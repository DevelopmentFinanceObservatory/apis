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

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain, groupby
from threading import Lock
from typing import TYPE_CHECKING, Literal, Protocol, cast, final

from dfo_spec_kit.units import Money, UnpricedMoney, parse_unit

from dc_graph import earliest_date as _earliest
from dc_graph import latest_date as _latest
from dc_sdmx import ContainedIn, EntityFilter
from dfo_platform.named import cut
from dfo_platform.problem import Alternative
from materialise.continued_series import ContinuedSeries
from materialise.errors import (
    CollectionProbeCeilingExceededError,
    DatasetDisagreementError,
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationUsageError,
    VariableBreadthUnreadError,
)
from materialise.graph import (
    Facet,
    FacetDefaultPolicy,
    GraphReader,
    GraphReadError,
    ProvenanceRefresh,
)
from materialise.method_status import status_of
from materialise.place_scope import (
    AvailabilityReader,
    DefaultPlaceScope,
    read_default_presence,
    scope_of_places,
)
from materialise.recipes import NonCurrencyUnitError
from materialise.window import RequestedWindow
from origin_catalogue import (
    UnUnserved,
    current_catalogue,
    record_un_read_failure,
    recording_un_blocked_places,
)
from origin_catalogue.graph import is_un_dcid
from origin_catalogue.origins import BASE, DFO
from series_coverage.coverage_types import BoundRoles, FacetCoverage, Role
from series_coverage.facets import FacetCandidate as _CoverageFacetCandidate
from series_coverage.facets import facet_candidates as _coverage_facet_candidates
from series_coverage.facets import rank_candidates as _coverage_rank_candidates
from series_coverage.schema import FacetMetadata, RoleNames

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
_PROVENANCE_REFRESH_PROPERTIES: tuple[str, ...] = ("lastDataRefreshDate",)

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


def unit_currency_code(ref: str | None) -> str | None:
    """The ISO 4217 code of a unit the unit registry reads as money
    (`dfo_spec_kit.units.parse_unit`: `CurrencyKES`, `USDollar`, a bare `INR`, `SDG_CU_USD`,
    `ChainedUSDollarBaseYr2017`), at any prices, or None for any other unit."""
    if ref is None:
        return None
    value = parse_unit(ref)
    return value.currency if isinstance(value, Money | UnpricedMoney) else None


def nominal_currency(ref: str | None) -> str | None:
    """The ISO 4217 code a native unit holds nominal amounts in, the currency a conversion
    starts from: a unit the registry reads as money at current or unstated prices
    (`CurrencyKES`, `USDollar`, a bare `INR`, `SDG_CU_USD`). A unit in constant prices
    (`undata/UNIT_MEASURE-CR_USD_K`, `ChainedUSDollarBaseYr2017`) holds none, and nor does any
    other unit, so a unit request over it raises `NonCurrencyUnitError`."""
    if ref is None:
        return None
    value = parse_unit(ref)
    if isinstance(value, UnpricedMoney) or (isinstance(value, Money) and value.prices == "current"):
        return value.currency
    return None


def currency_native_matches(candidates: Sequence[Facet], axis: str, currency: str) -> list[Facet]:
    """Candidates whose unit is denominated in `currency`: the `Currency<T>` ones when any
    exist, else those whose unit the registry reads as nominal money in it
    (`nominal_currency`)."""
    exact_ref = f"Currency{currency}"
    exact = [c for c in candidates if facet_axis_value(c, axis) == exact_ref]
    if exact:
        return exact
    return [c for c in candidates if nominal_currency(facet_axis_value(c, axis)) == currency]


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
class NationalCurrency:
    """The unit ask for each entity's own currency: at each entity `role` binds, the facets
    whose unit is the one currency the graph's `nationalCurrencyOf` names for it, served as
    published. Mirrors the identity core's `national_currency` member."""

    role: Literal["observationAbout", "origin", "destination"]


@dataclass(frozen=True)
class FacetBinding:
    """Binding for one `scope.facets` axis. `bound` carries a string `ref`, which callers
    must validate, or `None` for the facets that declare no value on the axis. Selection
    trusts it and matches candidates by `facet_axis_value` equality. A `request` binding
    carries `UnitRequest` and leaves `ref` unset, and a `national_currency` binding carries
    `NationalCurrency` alone. Both apply only to `unit`. `__post_init__` enforces consistent
    fields."""

    state: Literal["bound", "unbound", "request", "national_currency"]
    ref: str | None = None
    request: UnitRequest | None = None
    national_currency: NationalCurrency | None = None

    def __post_init__(self) -> None:
        if self.state in ("request", "national_currency") and self.ref is not None:
            raise MaterialisationUsageError(
                f"a {self.state}-state facet binding cannot carry a ref"
            )
        if (self.state == "request") != (self.request is not None):
            raise MaterialisationUsageError(
                "a facet binding carries a request exactly when its state is 'request', "
                f"got state {self.state!r}"
            )
        if (self.state == "national_currency") != (self.national_currency is not None):
            raise MaterialisationUsageError(
                "a facet binding carries national_currency exactly when its state is "
                f"'national_currency', got state {self.state!r}"
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
class RankedValue:
    """One value an axis election ranked, with the figures the ranking read for the
    best-ranked facet carrying it: `places`, its series count under the window (`breadth`),
    and `years`, the window years its span covers (`years_in_window`), each `None` when the
    election did not rank by it, and `observations`, its observation count."""

    ref: str | None
    places: int | None
    years: int | None
    observations: int

    def as_trace_entry(self) -> dict[str, object]:
        return {
            "ref": self.ref,
            "places": self.places,
            "years": self.years,
            "observations": self.observations,
        }


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
    # On the `measurementMethod` axis, the estimated method whose facets serve the window's
    # years past the selected series (`continuation`), or `None` when nothing continues it.
    continued_by: str | None = None
    # On the `unit` axis under a national-currency binding, the binding. The selection then
    # admits every unit (`admits`), `ref` is `None`, and the probe's place admission holds
    # each place to its own currency (`national_currency.narrow_to_national_currency`).
    national_currency: NationalCurrency | None = None
    # Under `most_observations_over_disagreement`, each distinct value on this axis in rank
    # order, the selected one first. Empty under every other reason.
    ranked_values: tuple[RankedValue, ...] = ()

    def admits(self, value: str | None) -> bool:
        """Whether a facet or row carrying `value` on this axis matches the selection: its
        ref, its `continued_by`, or any value under a national-currency binding."""
        return (
            self.national_currency is not None
            or value == self.ref
            or (self.continued_by is not None and value == self.continued_by)
        )

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
        if self.national_currency is not None:
            return {
                "axis": self.axis,
                "ref": {"national_currency": {"role": self.national_currency.role}},
                "basis": "bound",
            }
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

    @classmethod
    def of_trace_entry(
        cls, entry: Mapping[str, object], *, continued_by: str | None = None
    ) -> FacetSelection:
        """The selection a stored `trace.facets` entry (`as_trace_entry`) records, with the
        method `trace.continuations` names on its axis as `continued_by`. The entry keeps no
        unit `request`, `national_currency` or `ranked_values`. Of those `admits` reads only
        `national_currency`, which no group binds (`pipeline._refuse_group_national_currency`),
        so a group's rebuilt selection admits what its stored one did."""
        return cls(
            axis=cast(str, entry["axis"]),
            ref=cast("str | None", entry["ref"]),
            basis=cast(
                'Literal["bound", "declared_default", "deployment_fallback"]', entry["basis"]
            ),
            default_policy=cast("str | None", entry["default_policy"]),
            default_policy_version=cast("str | None", entry["default_policy_version"]),
            selection_reason=cast("str | None", entry["selection_reason"]),
            dispatch_facet_id=cast(str, entry["dispatch_facet_id"]),
            continued_by=continued_by,
        )

    def choice_entry(self) -> dict[str, object] | None:
        """The trace's `choices` entry for a value chosen over disagreeing candidates: `axis`
        and its `candidates` in rank order, the chosen value first. `None` when the selection
        chose over no disagreement."""
        if not self.ranked_values:
            return None
        return {
            "axis": self.axis,
            "candidates": [value.as_trace_entry() for value in self.ranked_values],
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
    year-grain dates, its first series' end to `first_series_end`, and its years with data
    (from the row year masks) to `dates`, so `overlaps_window` tests the points a window
    holds. Its `observation_period` comes from the coverage view's facet metadata, so
    `cadences_by_member` can tell a quarterly facet from an annual one whatever the year-grain
    dates say. Four-digit years sort lexically through
    9999. The coverage view's horizon starts in 1960."""
    return Facet(
        dates=tuple(str(year) for year in candidate.years),
        facet_id=candidate.facet_id,
        provenance_id=candidate.provenance_id,
        measurement_method=candidate.measurement_method,
        unit=candidate.unit,
        obs_count=candidate.n_obs,
        earliest_date=str(candidate.first_year),
        latest_date=str(candidate.last_year),
        first_series_end=(
            None
            if candidate.first_series_end in (None, candidate.last_year)
            else str(candidate.first_series_end)
        ),
        scaling_factor=candidate.scaling_factor,
        observation_period=candidate.observation_period,
        dates_coarsened=True,
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
    `[]` when it is covered but no rows match. It describes the rows the current request can
    serve; use ``role_bearing_facet_candidates`` to validate a replacement facet pin."""
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


def role_bearing_facet_candidates(
    *,
    stat_var: str,
    role_bindings: Mapping[str, EntityFilter],
    coverage: CoverageIndex,
    first_year: int | None = None,
    last_year: int | None = None,
) -> list[Facet] | None:
    """Facet choices that can validate a role-bearing request.

    Choices are globally catalogue-admitted at the bound role tuples, but do
    not inherit a current provenance pin or per-series precedence.  A choice
    can replace that pin, whereas ``role_bearing_facets`` describes rows the
    current request can actually serve.
    """
    bound = _bound_roles_for_coverage(role_bindings, coverage=coverage, stat_var=stat_var)
    if bound is None:
        return None
    return _coverage_facets(
        stat_var=stat_var,
        bound=bound,
        coverage=coverage,
        first_year=first_year,
        last_year=last_year,
        facet_candidates=True,
    )


def role_bearing_facet_candidates_of(
    *,
    stat_vars: Sequence[str],
    role_bindings: Mapping[str, EntityFilter],
    coverage: CoverageIndex,
    first_year: int | None = None,
    last_year: int | None = None,
) -> dict[str, list[Facet]]:
    """`role_bearing_facet_candidates` of each of `stat_vars`, read in one pass over the
    coverage index per distinct set of bound roles rather than once per variable.

    Each value equals `role_bearing_facet_candidates` for its variable, `[]` where that
    reads no row, in the order of `stat_vars`. A variable it answers `None` for, one without
    `RoleNames`, is left out."""
    groups: dict[frozenset[tuple[Role, frozenset[str]]], tuple[BoundRoles, list[str]]] = {}
    for stat_var in dict.fromkeys(stat_vars):
        bound = _bound_roles_for_coverage(role_bindings, coverage=coverage, stat_var=stat_var)
        if bound is not None:
            groups.setdefault(frozenset(bound.items()), (bound, []))[1].append(stat_var)
    found: dict[str, list[Facet]] = {}
    for bound, members in groups.values():
        facets_by_variable = {
            variable_coverage.variable: variable_coverage.facets
            for variable_coverage in coverage.filter_facet_candidates(
                variables=members, bound=bound, first_year=first_year, last_year=last_year
            )
        }
        for stat_var in members:
            found[stat_var] = _ranked_coverage_facets(
                facets_by_variable.get(stat_var, ()), metadata=coverage.facet_metadata
            )
    return {stat_var: found[stat_var] for stat_var in dict.fromkeys(stat_vars) if stat_var in found}


def role_bearing_facet_candidates_by_donor(
    *, stat_vars: Sequence[str], donors: Sequence[str], coverage: CoverageIndex
) -> dict[str, dict[str, list[Facet]]]:
    """Each of `stat_vars`' facet choices at each of `donors` bound alone to the donor role,
    the recipient open, read in one grouped pass over the variables' rows.

    Each value equals `role_bearing_facet_candidates` with only that donor bound, ranked the
    same way; a variable and donor with no rows map to `[]`. A variable lacking `RoleNames`,
    or declaring no donor role, holds no row at any donor, so every value it has is `[]`."""
    found = coverage.pair_facet_candidates(
        variables=stat_vars,
        bound={"donor": frozenset(donors)},
        pair_roles=("donor",),
        facet_candidates=True,
    )
    by_donor: dict[str, dict[str, list[Facet]]] = {
        stat_var: {donor: [] for donor in donors} for stat_var in stat_vars
    }
    for (stat_var, (donor,)), facet_coverage in found.items():
        candidates = _coverage_facet_candidates(facet_coverage, metadata=coverage.facet_metadata)
        ranked = _coverage_rank_candidates(candidates)
        by_donor[stat_var][donor] = [_facet_from_coverage_candidate(c) for c in ranked]
    return by_donor


def _coverage_facets(
    *,
    stat_var: str,
    bound: BoundRoles,
    coverage: CoverageIndex,
    first_year: int | None = None,
    last_year: int | None = None,
    facet_candidates: bool = False,
) -> list[Facet]:
    """`stat_var`'s served rows or pin-validation candidates, as ranked facets."""
    coverage_filter = (
        coverage.filter_facet_candidates if facet_candidates else coverage.filter_candidates
    )
    variable_coverage = coverage_filter(
        variables=[stat_var], bound=bound, first_year=first_year, last_year=last_year
    )
    facet_coverage = variable_coverage[0].facets if variable_coverage else ()
    return _ranked_coverage_facets(facet_coverage, metadata=coverage.facet_metadata)


def _ranked_coverage_facets(
    facet_coverage: Sequence[FacetCoverage], *, metadata: Mapping[str, FacetMetadata]
) -> list[Facet]:
    """One variable's facet coverage as ranked facets."""
    candidates = _coverage_facet_candidates(facet_coverage, metadata=metadata)
    return [_facet_from_coverage_candidate(c) for c in _coverage_rank_candidates(candidates)]


def coverage_probe(*, stat_var: str, bound: BoundRoles, coverage: CoverageIndex) -> MemberProbe:
    """A role-bearing member's probe from coverage under `bound`, in coverage's role labels:
    its candidates across every bound role tuple and every year, ranked
    (`_coverage_facets`), and each facet's series count for a window's years
    (`CoverageIndex.series_counts`), one series per donor and recipient pair holding data
    there. A mint probes here (`probe_facets`), and a Search read plan elects each member's
    group from the same probe, so both rank the same candidates by the same counts."""

    def breadth(window: RequestedWindow | None) -> SeriesCounts:
        first, last = (
            (window.start_year, window.end_year)
            if window is not None and window.kind == "range"
            else (None, None)
        )
        return SeriesCounts(
            counts=coverage.series_counts(
                variable=stat_var, bound=bound, first_year=first, last_year=last
            ),
            union_of=lambda ids: coverage.union_series_count(
                variable=stat_var, facet_ids=ids, bound=bound, first_year=first, last_year=last
            ),
        )

    admitted_pairs = coverage.admitted_pairs(variable=stat_var, bound=bound)
    admission = (
        CoverageAdmission(
            pairs_by_facet=admitted_pairs,
            role_names=coverage.role_names[stat_var],
        )
        if admitted_pairs is not None
        else CoverageAdmission()
    )
    return MemberProbe(
        candidates=tuple(_coverage_facets(stat_var=stat_var, bound=bound, coverage=coverage)),
        admission=admission,
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


class BreadthUnread(Enum):
    """A single-place probe's whole-variable breadth the reader could not read
    (`PendingBreadth.resolve`). Either way the tie skips it, and only a tie it would have
    decided acts on it (`rank_candidates`). Where several variables' breadths are summed, the
    first member in declaration order that any of them carries wins."""

    DEFECT = "defect"
    """For a caller storing its election, a Datasets mint, when the variable's own profile
    raised for a reason other than a read failing: the election refuses, and a retry would
    fail alike. A reader returns it for such a variable (`VariableBreadthReader`)."""
    UNREAD = "unread"
    """For a caller storing its election, when the read failed in a way a later read can
    answer (a timeout, a cap or graph read error, a refused flight): the election refuses,
    and a retry reads again."""
    FALLBACK = "fallback"
    """For a caller storing nothing, Search, whatever the cause: the tie falls to the unit's
    name, and the binding hears of it (`bound_variable_breadth`'s `on_fallback`)."""


type ReadBreadth = Mapping[str, int] | BreadthUnread
"""A whole-variable breadth once read: each facet id's places, or why it is missing."""


type VariableBreadth = Mapping[str, int] | BreadthUnread | PendingBreadth | None
"""Each facet id's places across a whole variable (`MemberProbe.variable_breadth`): a
`PendingBreadth` until a tie needs it, a `BreadthUnread` when the read failed, and None when
no breadth applies."""


type SeriesBreadth = Callable[[RequestedWindow | None], Mapping[str, int]]
"""Each facet id's series count under a window: the places, or the role tuples, at which
the facet holds a point in it. `rank_candidates` leads with it."""


@final
@dataclass(frozen=True, eq=False, slots=True)
class SeriesCounts(Mapping[str, int]):
    """Each facet id's series count (`counts`), read as a mapping, and the count of the series
    any of several facet ids holds (`union`), which a joined candidate ranks on
    (`_ranking_groups`). `union_of` runs only when the ranking forms a join, once per set of
    facet ids."""

    counts: Mapping[str, int]
    union_of: Callable[[frozenset[str]], int]
    _unions: dict[frozenset[str], int] = field(default_factory=dict, repr=False)

    def __getitem__(self, facet_id: str) -> int:
        return self.counts[facet_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self.counts)

    def __len__(self) -> int:
        return len(self.counts)

    def union(self, facet_ids: frozenset[str]) -> int:
        """The series at least one of `facet_ids` holds."""
        found = self._unions.get(facet_ids)
        if found is None:
            found = self._unions[facet_ids] = self.union_of(facet_ids)
        return found


def union_count(counts: Mapping[str, int], facet_ids: Iterable[str]) -> int:
    """The series at least one of `facet_ids` holds by `counts`: none, one facet's own count
    when only it holds any, and otherwise the `SeriesCounts` union of those that do. A facet
    `counts` omits holds none.

    Raises:
        TypeError: several of `facet_ids` hold series and `counts` is a plain mapping, which
            cannot tell how many series they share.
    """
    held = frozenset(f for f in facet_ids if counts.get(f, 0) > 0)
    if len(held) <= 1:
        return sum(counts[f] for f in held)
    if not isinstance(counts, SeriesCounts):
        raise TypeError(
            f"counting the series {sorted(held)} hold together needs SeriesCounts, "
            f"not {type(counts).__name__}"
        )
    return counts.union(held)


def summed_counts(members: Sequence[Mapping[str, int]]) -> SeriesCounts:
    """Several members' series counts summed, each member's series its own, so a union sums
    the members' unions."""
    total: dict[str, int] = {}
    for member in members:
        for facet_id, count in member.items():
            total[facet_id] = total.get(facet_id, 0) + count
    return SeriesCounts(
        counts=total, union_of=lambda ids: sum(union_count(member, ids) for member in members)
    )


def series_breadth(
    by_series: Mapping[str, Sequence[Facet]], *, window: RequestedWindow | None
) -> SeriesCounts:
    """How many of `by_series`' series each facet id holds a point at inside `window`
    (`overlaps_window`), and how many any of several does. A facet absent from the result
    holds none."""
    held_by_series = [
        frozenset(
            f.facet_id
            for f in facets
            if f.facet_id is not None and f.obs_count > 0 and overlaps_window(f, window=window)
        )
        for facets in by_series.values()
    ]
    counts: dict[str, int] = {}
    for held in held_by_series:
        for facet_id in held:
            counts[facet_id] = counts.get(facet_id, 0) + 1
    return SeriesCounts(
        counts=counts,
        union_of=lambda ids: sum(1 for held in held_by_series if not held.isdisjoint(ids)),
    )


@dataclass(frozen=True)
class PlaceAdmission:
    """The probed places at which origin admission passed each candidate facet id, for one
    subject-only StatVar. These are the places the request binds, or the default place scope's
    when it binds none. The graph probe reads admitted facets per place
    (`CatalogueSnapshot.admit_series`), and a fetched row serves only when its `facetId` was
    admitted at its `observationAbout` (`columnar.admission_mask`)."""

    places_by_facet: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class CoverageAdmission:
    """Admission for a role-bearing StatVar, whose candidates come from the coverage index.
    The unscoped sentinel has ``pairs_by_facet=None`` and serves every row. A source-scoped
    probe carries its admitted donor/recipient pairs per facet and the variable's graph role
    names, so an SDMX response cannot serve another pair holding the same facet."""

    pairs_by_facet: Mapping[str, frozenset[tuple[str, str]]] | None = None
    role_names: RoleNames | None = None


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
    unavailable (`place_scope.DefaultPlaceScope.every_place`). `variable_breadth` maps each
    facet id to the places, or role tuples, it holds a point at across the whole variable at
    its default place scope, and breaks a tie before the unit (`rank_candidates`). A probe of
    one bound place carries it as a `PendingBreadth` (`read_variable_breadths`), read only
    when a tie needs it, and every other probe carries None. `by_place` holds a subject-only
    probe's own facets at each probed place (`subject_probe`), and is empty for every other
    probe. `settled` is false when `places` are a countries fallback this request alone took
    (`place_scope.DefaultPlaceScope.settled`)."""

    candidates: tuple[Facet, ...]
    admission: RowAdmission
    places: tuple[str, ...] = ()
    every_place: bool = True
    breadth: SeriesBreadth | None = field(default=None, compare=False)
    variable_breadth: VariableBreadth = field(default=None, compare=False)
    by_place: Mapping[str, tuple[Facet, ...]] = field(default_factory=dict, compare=False)
    settled: bool = field(default=True, compare=False)

    def breadth_in(self, window: RequestedWindow | None) -> Mapping[str, int] | None:
        """Each facet id's series count under `window`, or None when the probe counts none."""
        return self.breadth(window) if self.breadth is not None else None

    @property
    def scope(self) -> DefaultPlaceScope:
        """`places`, `every_place` and `settled` as the default place scope that resolved
        them."""
        return DefaultPlaceScope(
            places=self.places, every_place=self.every_place, settled=self.settled
        )


# A subject-only member the probe read nothing for. It has no candidate and admits no row.
_NO_SUBJECT_PROBE = MemberProbe(candidates=(), admission=PlaceAdmission(places_by_facet={}))


class VariableBreadthReader(Protocol):
    """Reads each facet id's places, or role tuples, across each of `stat_vars` (bare dcids)
    at its default place scope. A variable whose breadth cannot be read in time is left out,
    and one whose own profile raised for a reason other than a read failing maps to
    `BreadthUnread.DEFECT`. `facet_profile.ProfileBreadth` reads it from the process's facet
    profiles."""

    def __call__(
        self, stat_vars: Sequence[str], *, deadline: float | None
    ) -> Mapping[str, Mapping[str, int] | BreadthUnread]: ...


class _BreadthReads:
    """One binding's whole-variable breadths: each variable read through `reader` at most
    once, its `PendingBreadth` shared by every probe of it, and its outcome kept, a failure
    included, so a request starts at most one profile flight per variable. Two threads
    missing the same variable at once both call `reader`, whose single-flight cache joins
    them to one flight."""

    def __init__(self, reader: VariableBreadthReader, *, refuse_unread: bool) -> None:
        self._reader = reader
        self._refuse_unread = refuse_unread
        self._lock = Lock()
        self._read: dict[str, ReadBreadth] = {}
        self._pending: dict[str, PendingBreadth] = {}

    def pending(self, stat_var: str, *, deadline: float | None) -> PendingBreadth:
        with self._lock:
            found = self._pending.get(stat_var)
            if found is None:
                found = self._pending[stat_var] = PendingBreadth(
                    stat_vars=(stat_var,), reads=self, deadline=deadline
                )
            return found

    def read(self, stat_vars: Sequence[str], *, deadline: float | None) -> list[ReadBreadth]:
        """`stat_vars`' breadths in order, reading the ones not read yet in one call."""
        with self._lock:
            missing = [sv for sv in dict.fromkeys(stat_vars) if sv not in self._read]
        if missing:
            found = self._reader(missing, deadline=deadline)
            with self._lock:
                for stat_var in missing:
                    self._read.setdefault(stat_var, self._outcome(found.get(stat_var)))
        with self._lock:
            return [self._read[sv] for sv in stat_vars]

    def _outcome(self, found: Mapping[str, int] | BreadthUnread | None) -> ReadBreadth:
        if isinstance(found, Mapping):
            return found
        if not self._refuse_unread:
            return BreadthUnread.FALLBACK
        return BreadthUnread.DEFECT if found is BreadthUnread.DEFECT else BreadthUnread.UNREAD


@final
@dataclass(frozen=True, eq=False, slots=True)
class PendingBreadth:
    """The whole breadth of `stat_vars`, summed across them, read through the bound reader
    the first time an election needs it (`resolve`). A tie needs it when the leading
    candidates tie on every earlier key (`rank_candidates`, `dispatch_cadence`); a probe whose
    election has no such tie reads nothing. It compares by identity, and every probe of a
    variable within one binding carries the same one."""

    stat_vars: tuple[str, ...]
    reads: _BreadthReads
    deadline: float | None

    def resolve(self) -> ReadBreadth:
        """The breadth summed across `stat_vars`, read once per variable per binding. A
        `BreadthUnread` when any variable's read failed, `DEFECT` ahead of `UNREAD`."""
        return _summed_breadth(self.reads.read(self.stat_vars, deadline=self.deadline))


def _summed_breadth(found: Sequence[ReadBreadth]) -> ReadBreadth:
    for unread in BreadthUnread:
        if unread in found:
            return unread
    if len(found) == 1:
        return found[0]
    read = [breadth for breadth in found if isinstance(breadth, Mapping)]
    assert len(read) == len(found)
    return summed_counts(read)


def joined_breadth(breadths: Sequence[VariableBreadth]) -> VariableBreadth:
    """Several variables' whole breadths summed, as a group's tie reads them
    (`group_facets.group_variable_breadth`): a `BreadthUnread` when any is one, `DEFECT`
    ahead of `UNREAD` ahead of `FALLBACK`; one `PendingBreadth` over every variable when each
    is pending under one binding, so the group reads them together and only on a tie; the sum
    when each is read; and None otherwise, so a member without one leaves the tie to the
    unit's name."""
    for unread in BreadthUnread:
        if any(breadth is unread for breadth in breadths):
            return unread
    if not breadths:
        return None
    pending = [b for b in breadths if isinstance(b, PendingBreadth)]
    if len(pending) == len(breadths) and len({id(b.reads) for b in pending}) == 1:
        return PendingBreadth(
            stat_vars=tuple(chain.from_iterable(b.stat_vars for b in pending)),
            reads=pending[0].reads,
            deadline=pending[0].deadline,
        )
    read = [b for b in breadths if isinstance(b, Mapping)]
    return _summed_breadth(read) if len(read) == len(breadths) else None


@dataclass(frozen=True, slots=True)
class _BreadthBinding:
    reads: _BreadthReads
    on_fallback: Callable[[], None] | None


_VARIABLE_BREADTH: ContextVar[_BreadthBinding | None] = ContextVar(
    "materialise_variable_breadth", default=None
)


@contextmanager
def bound_variable_breadth(
    reader: VariableBreadthReader | None,
    *,
    refuse_unread: bool,
    on_fallback: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Binds `reader` for the body of the `with` block, so a probe of one bound place carries
    its variable's whole breadth, read through it when a tie needs it
    (`read_variable_breadths`). A door binds its process-level reader around each request, as
    it binds the origin catalogue, and the binding reads each variable at most once
    (`PendingBreadth`). With none bound, a single-place tie falls to the unit's name.

    `refuse_unread` is for a caller that stores its election, a Datasets mint: an election
    that needs a breadth the reader could not answer raises `VariableBreadthUnreadError`
    (`rank_candidates`), transient for `BreadthUnread.UNREAD` and not for
    `BreadthUnread.DEFECT`. Without it, Search's case, the breadth is `BreadthUnread.FALLBACK`:
    an election that needs it falls to the unit's name and calls `on_fallback` in the context
    it ranks in. An election the breadth would not have decided reads nothing and does
    neither."""
    token = _VARIABLE_BREADTH.set(
        _BreadthBinding(
            reads=_BreadthReads(reader, refuse_unread=refuse_unread), on_fallback=on_fallback
        )
        if reader is not None
        else None
    )
    try:
        yield
    finally:
        _VARIABLE_BREADTH.reset(token)


def read_variable_breadths(
    stat_vars: Sequence[str], *, places: Sequence[str], deadline: float | None
) -> Mapping[str, PendingBreadth]:
    """Each of `stat_vars`' whole breadth (`MemberProbe.variable_breadth`), keyed by bare dcid,
    as a `PendingBreadth` that reads by `deadline` only when a tie needs it, when a request
    binds exactly one place, `places`, and a reader is bound (`bound_variable_breadth`). Empty
    otherwise, so a request naming several places has none. Nothing is read here."""
    binding = _VARIABLE_BREADTH.get()
    if binding is None or len(places) != 1:
        return {}
    return {
        bare: binding.reads.pending(bare, deadline=deadline)
        for bare in dict.fromkeys(sv.removeprefix(_STAT_VAR_REF_PREFIX) for sv in stat_vars)
    }


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
            "no probed facet satisfies the request's facet filter", facet_pointer="/facets"
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
    window: RequestedWindow | None = None,
    deadline: float | None = None,
) -> MemberProbe:
    """Probes candidates for the request's bound entities. The pipeline calls this once per
    generation and shares candidates across all three axes.

    Subject-only StatVars use one `graph.variable_facets_batch` call across all entities,
    which reads each facet's dates under a range `window` (`_reads_dates`), so the election
    judges a facet by the points it holds in the window (`overlaps_window`) as Search's dated
    read does.
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
    expansion and returns no candidates under an empty place admission. A probe of one bound
    place carries its variable's whole breadth (`read_variable_breadths`). Graph and
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
                    stat_var=stat_var,
                    entities=scope.places,
                    needs_dates=_reads_dates(window),
                    deadline=deadline,
                )
                if scope.places
                else {}
            )
            return _scoped_probe(by_entity, scope)
        by_entity = graph.variable_facets_batch(
            stat_var=stat_var,
            entities=bound_entities,
            needs_dates=_reads_dates(window),
            deadline=deadline,
        )
        breadths = read_variable_breadths([stat_var], places=bound_entities, deadline=deadline)
        return replace(
            subject_probe(by_entity, bound_entities),
            variable_breadth=breadths.get(stat_var.removeprefix(_STAT_VAR_REF_PREFIX)),
        )
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


def _reads_dates(window: RequestedWindow | None) -> bool:
    """Whether a subject-only facet probe under `window` reads each facet's dates: only under
    a range, the one window `overlaps_window` and `series_breadth` judge by dates. A facet
    whose dated read comes back short of its count (`Facet.dates_complete`) is judged by its
    span, as an undated one is."""
    return window is not None and window.kind == "range"


def _scoped_probe(
    by_entity: Mapping[str, Sequence[Facet]], scope: DefaultPlaceScope
) -> MemberProbe:
    """`subject_probe` at a default place scope's places, carrying the scope."""
    return replace(
        subject_probe(by_entity, scope.places),
        places=scope.places,
        every_place=scope.every_place,
        settled=scope.settled,
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
        by_place=by_place,
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
    every place with data, which holds only when each member's are, and settled only when
    each member's is. Empty when every member binds a place."""
    scope = scope_of_places(
        (place for member in members for place in member.places),
        every_place=all(member.every_place for member in members),
    )
    return replace(scope, settled=all(member.settled for member in members))


def merge_facets(facets: Sequence[Facet]) -> Facet:
    """The first of `facets` with the others' points added: counts summed, date bounds
    widened, dated points concatenated in order, the earliest series end kept
    (`Facet.first_series_end`), and annual evidence held only when every facet holds it
    (`Facet.reads_annual`). The merged dates are complete while
    each facet's dates account for its observations, so a facet read without dates leaves
    them incomplete. Metadata stays the first facet's.

    Each facet's points are copied once, so merging a facet across many series costs the
    points they hold, and a lone facet is returned as it is. `facets` is non-empty."""
    first = facets[0]
    if len(facets) == 1:
        return first
    dates = tuple(chain.from_iterable(f.dates for f in facets))
    latest = reduce(_latest, (f.latest_date for f in facets))
    series_end = reduce(_earliest, (f.earliest_series_end for f in facets))
    return replace(
        first,
        obs_count=sum(f.obs_count for f in facets),
        earliest_date=reduce(_earliest, (f.earliest_date for f in facets)),
        latest_date=latest,
        first_series_end=None if series_end == latest else series_end,
        dates=dates,
        values=tuple(chain.from_iterable(f.values for f in facets)),
        dates_complete=all(
            _dates_account_for_points(f) if dates else f.dates_complete for f in facets
        ),
        annual_evidence=all(f.reads_annual for f in facets),
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

    `provenance_refresh` maps distinct candidate provenance ids to their `lastDataRefreshDate`
    from one batched `node_arcs_many` call. It is empty when no candidate has an id. A run that
    reuses the probe takes its source tokens from it (`materialise.dependencies`).
    """

    members: tuple[MemberProbe, ...]
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
    deadline: float | None = None,
) -> CollectionFacetProbe:
    """Probes a request group and returns its candidates and provenance refresh dates. After the
    probe it fetches distinct provenance dates in one `node_arcs_many` call. That method has no
    `deadline` parameter. Empty `requests` returns empty candidates and refresh dates without
    network calls. For grouping and refusals, see `probe_collection_candidates`.

    The probe runs under its own `recording_un_blocked_places` record and returns what that
    record held as `un_unserved`. A UN read that fails beside a read of our variables
    therefore joins `un_unserved` and leaves the UN members without candidates."""
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
    The members of a group binding one place carry their variables' whole breadth, read once
    for the group (`read_variable_breadths`).

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
    bindings are skipped because expansion requires a graph read. A subject-only group's
    graph batches read dates when any of its members asks for a range window
    (`_reads_dates`), as `probe_facets` does.

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
        needs_dates = any(_reads_dates(requests[index].window) for index in indices)
        if binds_no_place(first.role_bindings, first.declared_roles):
            at_own_places = _probe_at_own_places(
                scopes, graph=graph, needs_dates=needs_dates, deadline=deadline
            )
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
                stat_vars=group_stat_vars,
                entities=group_entities,
                needs_dates=needs_dates,
                deadline=deadline,
            )
            breadths = read_variable_breadths(
                group_stat_vars, places=group_entities, deadline=deadline
            )
            for index in indices:
                bare_stat_var, bound_entities = member_specs[index]
                members_by_index[index] = replace(
                    subject_probe(by_stat_var.get(bare_stat_var, {}), bound_entities),
                    variable_breadth=breadths.get(bare_stat_var),
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
    scopes: Mapping[str, DefaultPlaceScope],
    *,
    graph: GraphReader,
    needs_dates: bool,
    deadline: float | None,
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
                    stat_vars=stat_vars,
                    entities=places,
                    needs_dates=needs_dates,
                    deadline=deadline,
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
    """Fetches each provenance's `lastDataRefreshDate` in one graph batch. A missing node or
    arc yields `ProvenanceRefresh(None)`, a source token with no date. Empty `provenance_ids`
    returns `{}` without a network call."""
    if not provenance_ids:
        return {}
    arcs_by_id = graph.node_arcs_many(provenance_ids, properties=_PROVENANCE_REFRESH_PROPERTIES)
    refresh_by_id: dict[str, ProvenanceRefresh] = {}
    for provenance_id in provenance_ids:
        arcs = arcs_by_id.get(provenance_id)
        last_values = arcs.get("lastDataRefreshDate", ()) if arcs is not None else ()
        refresh_by_id[provenance_id] = ProvenanceRefresh(
            last_data_refresh_date=last_values[0] if last_values else None
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
    variable_breadth: VariableBreadth = None,
    national_currency: bool = False,
) -> tuple[Facet, ...]:
    """Orders candidates holding a point in `window` first (`overlaps_window`), then those whose
    provenance is in `precedence` (`dfo_precedence`), then by descending `breadth`, descending
    window years covered (`years_in_window`), descending `obs_count`, descending
    `latest_date`, descending `variable_breadth`, then a method classed estimated or
    provisional after any other, from any source (`method_status.status_of`), then
    `provenance_id`, `measurement_method`, `unit`, and `facet_id`. A `None` tiebreak value
    sorts last. Stable sorts make the total order deterministic.

    `breadth` maps a `facet_id` to the number of series the facet holds a point at in
    `window`: the places, or the role tuples, a request binds (`MemberProbe.breadth`,
    `series_breadth`). One facet serves a variable across all of them, so the facet covering
    the most leads, ahead of the years and the count. A mint, a datasets read, a unit
    decision, a group selection and Search's read plans, scope and bulk elections all pass
    it. A facet it omits counts zero. `None`, for a single series, skips the step.

    `variable_breadth` maps a `facet_id` to the places, or role tuples, it holds a point at
    across the whole variable at its default place scope (`MemberProbe.variable_breadth`),
    which a probe carries only when the request binds one place. There every facet's
    `breadth` is one, so a facet that ties on years, count and latest date goes to the one
    serving the variable most widely, as a request naming many places would elect, before
    the unit's name decides: France alone gets WEO's US-dollar GDP (197 places) over its euro
    series (25). A facet it omits counts zero. `None` skips the step. A `PendingBreadth` is
    read only when the leading candidates tie on every earlier key and differ in provenance,
    method or unit, so the breadth decides the axes (`_axes_breadth_decides`); otherwise it
    reads nothing, since the breadth would not change the top. A `BreadthUnread`, or a read
    that fails, skips the step and acts on that tie (`_act_on_unread`). `dispatch_cadence`
    does the same for the cadence.

    `national_currency` is true under a unit binding to each place's national currency, where
    a candidate counts the series of every facet sharing its provenance, method and cadence
    (`_ranking_groups`): one provenance's facets each hold the places of one currency, and
    together they serve all of them.

    `window` is required so every ranking site states whether a requested year range applies.
    Under a range the election is the top span-ranked facet among those known to hold a
    point in the window: the facet whose span covers more of the window's years leads, so
    one spanning 2020 to 2022 ranks above one spanning 1970 to 2021 for a 2020 to 2022
    request, and lifetime count breaks a tie. `None`, `all` and `latest` windows rank by
    count. The top candidate is the top of `narrow_to_window`'s pool.

    Every probe that elects under a range reads its candidates' dates (Search's answer and
    option-coverage reads, a mint's `probe_facets` and `probe_collection_candidates`,
    coverage's years), so a facet
    with a gap across the whole window leaves the pool on every side. A candidate whose dates
    are incomplete (`Facet.dates_complete`) is judged by its span."""
    groups = _ranking_groups(
        candidates, window=window, breadth=breadth, national_currency=national_currency
    )
    return _ranked_facets(
        _rank_groups(
            groups,
            window=window,
            precedence=precedence,
            breadth=breadth,
            variable_breadth=variable_breadth,
            national_currency=national_currency,
        )
    )


def _rank_groups(
    groups: Sequence[_RankingGroup],
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
    variable_breadth: VariableBreadth,
    national_currency: bool,
) -> list[_RankingGroup]:
    """`groups` (`_ranking_groups`) in `rank_candidates`' order, reading `variable_breadth`
    only when it decides a tie."""
    breadth = _joined_counts(groups, breadth)
    rank = partial(_ranked, groups, window=window, precedence=precedence, breadth=breadth)
    if not isinstance(variable_breadth, PendingBreadth | BreadthUnread):
        return rank(_joined_counts(groups, variable_breadth))
    ranked = rank(None)
    decided = _axes_breadth_decides(
        ranked,
        window=window,
        precedence=precedence,
        breadth=breadth,
        national_currency=national_currency,
    )
    if decided:
        read = _read_breadth(variable_breadth)
        if isinstance(read, Mapping):
            return rank(_joined_counts(groups, read))
        _act_on_unread(read, decided)
    return ranked


@dataclass(frozen=True)
class _RankingGroup:
    """One candidate as the ranking sees it: a facet (`lead`), the estimated facets that
    continue its selection's series (`continuing`, `_ranking_groups`), `joined`, the lead
    merged with them (`merge_facets`), whose span, count and dates the ranking reads, and
    `counted`, the facet ids whose series it counts together (`_joined_counts`)."""

    lead: Facet
    continuing: tuple[Facet, ...]
    joined: Facet
    counted: frozenset[str]


type _SelectionKey = tuple[str | None, str | None, str | None, Cadence | None]
"""A selection's provenance, method, unit and cadence (`_selection_key`)."""


def _selection_key(
    facet: Facet, *, cadence_of: Mapping[str, Cadence], national_currency: bool
) -> _SelectionKey:
    """The selection `facet` belongs to: the provenance, method and unit an election taking
    it as its top would select, and the cadence its dispatch would serve. Under a
    national-currency unit binding the unit is each place's own, so it leaves the key."""
    return (
        facet.provenance_id,
        facet.measurement_method,
        None if national_currency else facet.unit,
        cadence_of.get(facet.facet_id or ""),
    )


def _ranking_groups(
    candidates: Sequence[Facet],
    *,
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None = None,
    national_currency: bool = False,
) -> list[_RankingGroup]:
    """`candidates` as the candidates the ranking compares under `window`, in candidate
    order, each formed alike in any order.

    A selection is the facets of `candidates` that an election taking any of them as its top
    would dispatch together: those sharing provenance, method, unit and cadence
    (`_selection_key`). Its continuation is the one a dispatch serves (`continuation` over
    the whole selection, with `breadth` choosing the method), so it continues each series
    the selection holds past that series' own end, as `continued_election`
    continues a served set. Each of the selection's facets that holds a point in `window` then ranks
    as one candidate with those estimated facets, by the years, count and latest date of
    both, and such an estimated facet never ranks on its own. One that continues several
    selections joins each of them, since each selection's dispatch would serve it, so the
    candidates compare whatever their order. Every other facet, and every facet under a
    window without an end year, is a candidate of its own, so a ranking where nothing
    continues is unchanged.

    A candidate counts the series of its lead and continuation together. Under a
    national-currency unit binding (`national_currency`) it counts those of its whole
    selection too: each place holds only facets in its own currency
    (`national_currency.narrow_to_national_currency`), so a provenance and method serve
    the places of all their facets, whatever the unit."""
    (cadence_of,) = cadences_by_member([candidates])
    key = partial(_selection_key, cadence_of=cadence_of, national_currency=national_currency)
    selections: dict[_SelectionKey, list[Facet]] = {}
    for candidate in candidates:
        if candidate.facet_id is not None:
            selections.setdefault(key(candidate), []).append(candidate)
    continuing: dict[_SelectionKey, tuple[Facet, ...]] = {}
    joined_to: set[int] = set()
    if window is not None and window.kind == "range" and window.end_year is not None:
        for selection_key, members in selections.items():
            if not any(overlaps_window(c, window=window) for c in members):
                continue
            serves = partial(at_cadence, cadence=selection_key[-1], cadence_of=cadence_of)
            rest = continuation(members, candidates, window=window, serves=serves, breadth=breadth)
            continuing[selection_key] = rest
            joined_to.update(id(c) for c in rest)
    groups: list[_RankingGroup] = []
    for candidate in candidates:
        if id(candidate) in joined_to:
            continue
        selection_key = key(candidate)
        rest = (
            continuing.get(selection_key, ())
            if candidate.facet_id is not None and overlaps_window(candidate, window=window)
            else ()
        )
        own = selections.get(selection_key, []) if national_currency else [candidate]
        groups.append(
            _RankingGroup(
                lead=candidate,
                continuing=rest,
                joined=merge_facets([candidate, *rest]) if rest else candidate,
                counted=frozenset(f.facet_id for f in (*own, *rest) if f.facet_id is not None),
            )
        )
    return groups


def _joined_counts(
    groups: Sequence[_RankingGroup], counts: Mapping[str, int] | None
) -> Mapping[str, int] | None:
    """`counts` (a `breadth` or a read whole-variable breadth) with each candidate's lead
    counting the places or series any of its counted facets holds (`union_count`), since
    the candidate serves each of them. Only a candidate counting several facets counts a
    union."""
    if counts is None or not any(len(group.counted) > 1 for group in groups):
        return counts
    joined = dict(counts)
    for group in groups:
        if len(group.counted) > 1 and group.lead.facet_id is not None:
            joined[group.lead.facet_id] = union_count(counts, group.counted)
    return joined


def _ranked_facets(ranked: Sequence[_RankingGroup]) -> tuple[Facet, ...]:
    """The facets of `ranked`, each candidate's lead followed by its continuing facets, each
    facet once: the leads of one selection share its continuation, which follows the first."""
    seen: set[int] = set()
    facets: list[Facet] = []
    for group in ranked:
        for facet in (group.lead, *group.continuing):
            if id(facet) not in seen:
                seen.add(id(facet))
                facets.append(facet)
    return tuple(facets)


def _ranked(
    groups: Sequence[_RankingGroup],
    variable_breadth: Mapping[str, int] | None,
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
) -> list[_RankingGroup]:
    """`groups` (`_ranking_groups`) in `rank_candidates`' total order, each ranking on its
    joined facet, under `breadth` and a read `variable_breadth` already joined
    (`_joined_counts`)."""
    ranked = list(groups)
    ranked.sort(key=lambda g: _none_last(g.joined.facet_id))
    ranked.sort(key=lambda g: _none_last(g.joined.unit))
    ranked.sort(key=lambda g: _none_last(g.joined.measurement_method))
    ranked.sort(key=lambda g: _none_last(g.joined.provenance_id))
    ranked.sort(key=lambda g: _estimated_last(g.joined))
    if variable_breadth is not None:
        ranked.sort(key=lambda g: variable_breadth.get(g.joined.facet_id or "", 0), reverse=True)
    # `latest_date` sorts descending (most recent first) with `None` last:
    # treating `None` as `""`, the empty string, sorts lowest in ascending
    # order and therefore last once the whole pass is reversed.
    ranked.sort(key=lambda g: g.joined.latest_date or "", reverse=True)
    ranked.sort(key=lambda g: g.joined.obs_count, reverse=True)
    ranked.sort(key=lambda g: years_in_window(g.joined, window=window), reverse=True)
    if breadth is not None:
        ranked.sort(key=lambda g: breadth.get(g.joined.facet_id or "", 0), reverse=True)
    ranked.sort(key=lambda g: g.joined.provenance_id not in precedence)
    ranked.sort(key=lambda g: not overlaps_window(g.joined, window=window))
    return ranked


def _estimated_last(facet: Facet) -> bool:
    """True when `facet`'s method is classed estimated or provisional (`status_of`), so it
    ranks after a facet with any other method, no method included."""
    return status_of(facet.measurement_method, facet.provenance_id) in ("estimated", "provisional")


def _read_breadth(variable_breadth: PendingBreadth | BreadthUnread) -> ReadBreadth:
    """`variable_breadth` read, for a tie it decides."""
    if isinstance(variable_breadth, PendingBreadth):
        return variable_breadth.resolve()
    return variable_breadth


def _leading_keys(
    facet: Facet,
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
) -> tuple[object, ...]:
    """The keys `rank_candidates` sorts on before whole-variable breadth. Two facets tie when
    their keys are equal, and tied facets sit together in the ranking."""
    return (
        overlaps_window(facet, window=window),
        facet.provenance_id in precedence,
        breadth.get(facet.facet_id or "", 0) if breadth is not None else 0,
        years_in_window(facet, window=window),
        facet.obs_count,
        facet.latest_date or "",
    )


def _axes_breadth_decides(
    ranked: Sequence[_RankingGroup],
    *,
    window: RequestedWindow | None,
    precedence: frozenset[str],
    breadth: Mapping[str, int] | None,
    national_currency: bool,
) -> list[str]:
    """The facet ids of the leading candidates of `ranked` (`_ranked`, the groups the
    ranking formed, under `breadth` joined as it ranked them), when they tie on every key
    before whole-variable breadth (`_leading_keys`) and differ in provenance, method or unit,
    the axes an election takes from its top facet, so the breadth would have decided them.
    Under a national-currency unit binding the unit is each place's own, so only provenance
    and method count. Empty otherwise. Which cadence a dispatch serves among facets carrying
    those axes is `dispatch_cadence`'s decision (`_cadence_breadth_decides`)."""
    if not ranked:
        return []

    def key(c: Facet) -> tuple[object, ...]:
        return _leading_keys(c, window=window, precedence=precedence, breadth=breadth)

    joined = [group.joined for group in ranked]
    lead = key(joined[0])
    tied = [c for c in joined if key(c) == lead]
    axes = {
        (c.provenance_id, c.measurement_method, None if national_currency else c.unit) for c in tied
    }
    return [c.facet_id or "" for c in tied] if len(axes) > 1 else []


def _act_on_unread(variable_breadth: BreadthUnread, decided_by_breadth: Sequence[str]) -> None:
    """When the unread breadth would have decided between `decided_by_breadth`'s facets,
    `BreadthUnread.UNREAD` raises a transient `VariableBreadthUnreadError`,
    `BreadthUnread.DEFECT` one that is not, and `BreadthUnread.FALLBACK` calls the bound
    `on_fallback` (`bound_variable_breadth`). Nothing happens otherwise."""
    if not decided_by_breadth:
        return
    if variable_breadth is not BreadthUnread.FALLBACK:
        raise VariableBreadthUnreadError(
            decided_by_breadth, transient=variable_breadth is BreadthUnread.UNREAD
        )
    binding = _VARIABLE_BREADTH.get()
    if binding is not None and binding.on_fallback is not None:
        binding.on_fallback()


@dataclass(frozen=True)
class RankedPool:
    """The fallback pool (`pool`, `fallback_pool`), the `dfo_precedence` set read on that pool,
    the pool in deployment-policy order (`ranked`), top first, and the candidates the
    ranking compared (`groups`, `_ranking_groups`), which `ranked_values` reports."""

    pool: tuple[Facet, ...]
    precedence: frozenset[str]
    ranked: tuple[Facet, ...]
    groups: tuple[_RankingGroup, ...]


def rank_pool(
    candidates: Sequence[Facet],
    *,
    stat_vars: Iterable[str],
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None = None,
    declared: Mapping[str, str | None] | None = None,
    variable_breadth: VariableBreadth = None,
    national_currency: bool = False,
) -> RankedPool:
    """Keeps the `candidates` carrying every declared default in `declared` and narrows them
    to `window` (`fallback_pool`), then ranks them with DFO's first for a variable in the
    origin catalogue's overlap set when DFO's facet is at least as broad as any other
    (`dfo_precedence`), by `breadth`, by the window years each span covers and by count
    (`rank_candidates`, `years_in_window`), with `variable_breadth` breaking a tie that
    remains before the unit. `national_currency` counts each candidate as `rank_candidates`
    does under a national-currency unit binding.

    This is the one deployment-fallback election. A single member (`select_facet`), a unit
    request (`_select_unit_request`, `decide_member_unit`), a group (`select_group_facets`)
    and Search's read plan, previews and bulk elections rank their pools here over each
    candidate's span and count, the fields every probe carries, after the pool drops the
    facets whose dates hold no point in a range window (`overlaps_window`). A search answer
    and a mint or datasets read of its Spec under the same window therefore serve the same
    facet."""
    pool = tuple(fallback_pool(candidates, declared or {}, window=window))
    groups = _ranking_groups(
        pool, window=window, breadth=breadth, national_currency=national_currency
    )
    precedence = dfo_precedence(stat_vars, pool, groups=groups, breadth=breadth)
    ranked = _rank_groups(
        groups,
        window=window,
        precedence=precedence,
        breadth=breadth,
        variable_breadth=variable_breadth,
        national_currency=national_currency,
    )
    return RankedPool(
        pool=pool, precedence=precedence, ranked=_ranked_facets(ranked), groups=tuple(groups)
    )


def continuation(
    served: Sequence[Facet],
    candidates: Sequence[Facet],
    *,
    window: RequestedWindow | None,
    serves: Callable[[str], bool],
    breadth: Mapping[str, int] | None = None,
) -> tuple[Facet, ...]:
    """The facets of `candidates` that continue the `served` series past their last years,
    or none.

    Each series continues past its own end, so a range window whose end year runs past the
    end of any series the `served` facets hold, at any place or role tuple
    (`Facet.earliest_series_end`), is continued by the facets that share the served
    provenance and a served unit, sit at the served cadence (`serves`, by facet id), hold
    estimated values (`method_status.status_of`), reach a later year than that series and
    hold a point in the window. When several estimated methods qualify, the one whose facet
    ranks first (`rank_candidates`) serves. A served series that is itself estimated
    continues with nothing, as do served series that all reach the window's end and any
    window without an end year. The ranking joins a candidate to the same facets
    (`_ranking_groups`), so it counts only what a dispatch serves.

    The continuation dispatches beside the served series, and each row keeps its own
    `measurementMethod`. A served value wins any period both hold, and estimated rows
    serve only the periods after a series' last served one
    (`columnar.continuation_mask`), so a series reaching the window's end takes none."""
    if not served:
        return ()
    head = served[0]
    ends = [y for f in served if (y := _leading_year(f.earliest_series_end)) is not None]
    if not ends:
        return ()
    continuing = continuing_candidates(
        candidates,
        provenance=head.provenance_id,
        method=head.measurement_method,
        units={f.unit for f in served},
        after=min(ends),
        window=window,
        serves=serves,
    )
    if not continuing:
        return ()
    method = rank_candidates(continuing, window=window, breadth=breadth)[0].measurement_method
    return tuple(c for c in continuing if c.measurement_method == method)


def continuing_candidates(
    candidates: Sequence[Facet],
    *,
    provenance: str | None,
    method: str | None,
    units: set[str | None],
    after: int | None,
    window: RequestedWindow | None,
    serves: Callable[[str], bool],
) -> list[Facet]:
    """The `candidates` that can continue a series of `method` under `provenance` in one of
    `units` that ends in the year `after` (`continuation`), in candidate order: those that
    share the provenance and a unit, sit at the served cadence (`serves`, by facet id), hold
    estimated values (`method_status.status_of`), reach a later year than `after` and hold a
    point in the window. An empty list under a window that is not a range with an end year,
    for an estimated `method`, or when `after` reaches the window's end. The units share one value
    except under a national-currency binding, where each place's admission keeps its own.
    `after` None sets no lower bound, for a series whose end is unknown."""
    if window is None or window.kind != "range" or window.end_year is None:
        return []
    if status_of(method, provenance) == "estimated":
        return []
    if after is not None and after >= window.end_year:
        return []
    return [
        c
        for c in candidates
        if c.provenance_id == provenance
        and c.unit in units
        and status_of(c.measurement_method, c.provenance_id) == "estimated"
        and (after is None or (_leading_year(c.latest_date) or 0) > after)
        and c.facet_id is not None
        and serves(c.facet_id)
        and overlaps_window(c, window=window)
    ]


def continues_selection(selection: FacetSelection) -> bool:
    """Whether a `measurementMethod` selection can be continued (`continued_election`): one
    the request does not bind."""
    return selection.basis != "bound"


def continued_election(
    election: FacetElection,
    members: Sequence[Sequence[Facet]],
    *,
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None = None,
) -> FacetElection:
    """`election` with its `measurementMethod` selection's `continued_by` naming the method
    that continues its served series (`continuation`). `members` holds each member's
    candidates in election order: one for a member's own election, one per member for a
    group's. A facet counts for a member only at the election's cadence
    (`FacetElection.serves`), and the members' copies of a facet id are merged
    (`merge_facets`), so a group continues as a request naming all its places does. An
    election whose method axis the request binds, or whose series nothing continues, is
    returned as it is. `joint_matches`, and so every member's dispatch, then admits the
    continuing facets beside the served ones."""
    selection = election.selections.get("measurementMethod")
    if selection is None or not continues_selection(selection):
        return election
    by_id: dict[str, list[Facet]] = {}
    for member, candidates in enumerate(members):
        for c in candidates:
            if c.facet_id is not None and election.serves(c.facet_id, member=member):
                by_id.setdefault(c.facet_id, []).append(c)
    held = [merge_facets(copies) for copies in by_id.values()]
    continuing = continuation(
        joint_matches(held, election.selections),
        held,
        window=window,
        serves=lambda _facet_id: True,
        breadth=breadth,
    )
    if not continuing:
        return election
    continued = replace(selection, continued_by=continuing[0].measurement_method)
    return replace(election, selections={**election.selections, "measurementMethod": continued})


def selected_continued_series(
    selections: Mapping[str, FacetSelection],
) -> tuple[ContinuedSeries, ...]:
    """The continuation of the series `selections` serve: the method continuing them
    (`FacetSelection.continued_by`), scoped to the method and source they are held under, or
    nothing when nothing continues them. It is the entry `continued_series_of_trace` reads
    back from the trace the same selections write."""
    method = selections.get("measurementMethod")
    if method is None or method.continued_by is None:
        return ()
    provenance = selections.get("provenance")
    return (
        ContinuedSeries(
            method=method.continued_by,
            continues=method.ref,
            provenance=None if provenance is None else provenance.ref,
        ),
    )


def dfo_precedence(
    stat_vars: Iterable[str],
    candidates: Sequence[Facet],
    *,
    groups: Sequence[_RankingGroup],
    breadth: Mapping[str, int] | None = None,
) -> frozenset[str]:
    """DFO's provenances among `candidates` when the bound origin catalogue lists any of
    `stat_vars` under both DFO and base Data Commons (its overlap set) and DFO's broadest
    candidate holds at least as many of the request's series as the broadest candidate of
    any other provenance, and empty otherwise. Each candidate is measured as
    `rank_candidates` reads it: a candidate of `groups` (`_ranking_groups` over
    `candidates`) by the series any of its counted facets holds (`_joined_counts`), so
    DFO's reported series and the estimates continuing it count together.

    Admission withholds no base facet at DFO's places, so whichever origin the election
    picks serves every place it holds, and one facet group serves the request. DFO's facet
    leads when it covers as many places as the best other facet; a narrower DFO facet gives
    way, so a population request for every country serves the base source's countries.
    `None` breadth, for a single series, gives every facet the same breadth and DFO leads. Outside the overlap set every candidate comes from one origin and
    the ranking is unchanged."""
    catalogue = current_catalogue()
    if not any(
        DFO in (held := catalogue.data_commons(stat_var.removeprefix(_STAT_VAR_REF_PREFIX)))
        and BASE in held
        for stat_var in stat_vars
    ):
        return frozenset()
    dfo = frozenset(
        provenance_id
        for candidate in candidates
        if (provenance_id := candidate.provenance_id) is not None
        and (row := catalogue.provenance(provenance_id)) is not None
        and row.data_commons == DFO
    )
    if breadth is None or not dfo:
        return dfo
    joined = _joined_counts(groups, breadth) or {}
    widest = {True: 0, False: 0}
    for group in groups:
        is_dfo = group.lead.provenance_id in dfo
        widest[is_dfo] = max(widest[is_dfo], joined.get(group.lead.facet_id or "", 0))
    return dfo if widest[True] >= widest[False] else frozenset()


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
    variable_breadth: VariableBreadth = None,
    requires_provenance: bool = True,
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
    facet's series count under `window` (`MemberProbe.breadth_in`), with `variable_breadth`
    breaking a tie before the unit (`rank_candidates`). An empty pool raises
    `DefaultFacetUndefinedError`. A bound ref or declared default ignores `window`.

    `policies` (`declared_default_policies`) holds every unresolved axis's declared default,
    read once for all axes. This axis's policy comes from the map, and a fallback pool keeps
    only the facets carrying the other axes' declared defaults (`fallback_pool`), so a
    declared default wins before window narrowing on every axis.

    Fallback records `sole_candidate`, `most_observations`, `dfo_precedence` when
    `dfo_precedence` ranked DFO's facet over another origin's, or
    `most_observations_over_disagreement`, which triggers a warning. Each reason describes the
    window-narrowed pool. About two in five observed pairs lack `measurementMethod` or `unit`,
    so a missing value returns `ref=None` on those axes. Missing provenance returns
    `ref=None` too, unless `requires_provenance`
    (`pipeline.MaterialisationRequest.requires_provenance`), when it raises
    `DefaultFacetUndefinedError`. The caller checks cross-axis fit."""
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
            variable_breadth=variable_breadth,
        )

    if binding.state == "national_currency":
        assert binding.national_currency is not None  # `FacetBinding.__post_init__`.
        return _select_national_currency(
            axis, binding.national_currency, dispatchable=dispatchable, all_bindings=all_bindings
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
            facet_pointer=f"/facets/{axis}",
            alternatives=alternatives,
            alternatives_truncated=alternatives_truncated,
        )

    viable = _narrow_by_bound_axes(dispatchable, all_bindings)
    if not viable:
        raise DefaultFacetUndefinedError(
            f"facet axis {axis!r} is unbound, no default policy is declared, and no "
            "probed candidate has any observations",
            facet_pointer=f"/facets/{axis}",
        )
    declared = _declared_refs(policies, dispatchable, all_bindings, except_axis=axis)
    national_currency = binds_national_currency(all_bindings)
    ranked_pool = rank_pool(
        viable,
        stat_vars=[stat_var],
        window=window,
        declared=declared,
        breadth=breadth,
        variable_breadth=variable_breadth,
        national_currency=national_currency,
    )
    viable, precedence, top = ranked_pool.pool, ranked_pool.precedence, ranked_pool.ranked[0]
    value = facet_axis_value(top, axis)
    if value is None:
        if axis == "provenance" and requires_provenance:
            raise DefaultFacetUndefinedError(
                "facet axis 'provenance' is unbound, no default policy is declared, and "
                "the winning candidate carries no provenance",
                facet_pointer="/facets/provenance",
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
        ranked_values=ranked_values(
            ranked_pool, axis, reason=reason, window=window, breadth=breadth
        ),
    )


def binds_national_currency(bindings: Mapping[str, FacetBinding]) -> bool:
    """Whether `bindings` bind the unit axis to each place's national currency, under which
    the ranking counts each provenance and method across all its units (`rank_candidates`)."""
    unit = bindings.get("unit")
    return unit is not None and unit.state == "national_currency"


def _select_national_currency(
    axis: str,
    national_currency: NationalCurrency,
    *,
    dispatchable: Sequence[Facet],
    all_bindings: Mapping[str, FacetBinding],
) -> FacetSelection:
    """Resolves a unit axis bound to each place's national currency. The candidates hold
    only facets in their places' own currencies (`national_currency.narrow_to_national_currency`),
    so the selection admits every unit they carry, with `ref` `None` and `basis="bound"`;
    the other axes elect over the same candidates. Its witness is the first candidate the
    bound axes admit, which `facet_election` re-points to the election's top. Raises
    `MaterialisationUsageError` off the unit axis, and `FacetNotAvailableError` when no
    candidate the bound axes admit remains."""
    if axis != "unit":
        raise MaterialisationUsageError(
            f"a national_currency facet binding only applies to the unit axis, not {axis!r}"
        )
    viable = _narrow_by_bound_axes_only(dispatchable, all_bindings)
    if not viable:
        raise FacetNotAvailableError(
            "facet axis 'unit' binds each place's national currency, and no candidate the "
            "bound axes admit holds observations in one",
            facet_pointer="/facets/unit",
        )
    return replace(
        _selection(
            axis,
            None,
            basis="bound",
            dispatch_facet_id=viable[0].facet_id,
            selection_reason="national_currency",
        ),
        national_currency=national_currency,
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
    bound_to = repr(cut(ref)) if ref is not None else "no declared value"
    raise FacetNotAvailableError(
        f"facet axis {axis!r} bound to {bound_to}, which has no observations",
        facet_pointer=f"/facets/{axis}",
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
    variable_breadth: VariableBreadth,
) -> FacetSelection:
    """Resolves a unit request from candidates narrowed by every bound axis, then by the other
    axes' declared defaults and `window` (`fallback_pool`). This shared pool keeps unit
    selection consistent with sibling fallback choices. A declared default is matched before
    the window narrows the pool, as `select_facet` ignores `window` for a declared default.

    An empty pool raises `FacetNotAvailableError` with alternatives from the original
    dispatchable candidates.

    A current-price ISO request prefers a candidate denominated in that currency. Otherwise
    the declared or ranked default applies. The result must be a currency ref
    (`nominal_currency`). Missing or other refs raise `NonCurrencyUnitError`
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
            facet_pointer=f"/facets/{axis}",
            alternatives=alternatives,
            alternatives_truncated=alternatives_truncated,
        )
    declared = _declared_refs(policies, dispatchable, all_bindings, except_axis=axis)
    ranked_pool = rank_pool(
        viable,
        stat_vars=[stat_var],
        window=window,
        declared=declared,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    in_window, precedence = ranked_pool.pool, ranked_pool.precedence
    selection = _preferred_native_unit_selection(
        axis,
        request,
        in_window,
        window=window,
        precedence=precedence,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )
    if selection is None:
        selection = _select_unit_request_default(
            axis,
            viable,
            ranked_pool,
            request,
            policy=policies.get(axis),
            window=window,
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
    variable_breadth: VariableBreadth,
) -> FacetSelection | None:
    """For a current-price ISO request, selects the top-ranked candidate denominated in
    `T`, preferring `Currency<T>` over a `CurrencyUnitOfMeasure` unit such as `USDollar`.
    Returns `None` for role currencies, constant prices, or no match, so the caller uses the
    default. Shared by member and group selection."""
    if request.prices != "current" or not _is_iso_currency(request.currency):
        return None
    matching = currency_native_matches(candidates, axis, request.currency)
    if not matching:
        return None
    top = rank_candidates(
        matching,
        window=window,
        precedence=precedence,
        breadth=breadth,
        variable_breadth=variable_breadth,
    )[0]
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
    ranked_pool: RankedPool,
    request: UnitRequest,
    *,
    policy: FacetDefaultPolicy | None,
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None,
) -> FacetSelection:
    """Selects the declared `policy` from `viable`, narrowed by bound axes, or else the
    top of `ranked_pool`, `viable`'s fallback pool ranked with its precedence. Every result
    carries the `UnitRequest`."""
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
    in_window, precedence = ranked_pool.pool, ranked_pool.precedence
    top = ranked_pool.ranked[0]
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
        ranked_values=ranked_values(
            ranked_pool, axis, reason=reason, window=window, breadth=breadth
        ),
    )


def _require_currency_native_unit(ref: str | None) -> None:
    """Raises `NonCurrencyUnitError` for a ref `nominal_currency` does not map to a currency.
    Live `CurrencyUnit` validation happens downstream without another graph read here."""
    if nominal_currency(ref) is not None:
        return
    raise NonCurrencyUnitError(
        f"unit request resolved to native unit {ref!r}, which is not a currency",
        unit=ref if ref is not None else "",
        facet_pointer="/facets/unit",
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
    currency when any exist (`currency_native_matches`). Otherwise returns `viable`
    unchanged. Groups apply this after member pools intersect."""
    unit_binding = all_bindings.get("unit", FacetBinding(state="unbound"))
    if unit_binding.state == "request":
        request = unit_binding.request
        assert request is not None  # `FacetBinding.__post_init__` guarantees this.
        if request.prices == "current" and _is_iso_currency(request.currency):
            currency_matched = currency_native_matches(viable, "unit", request.currency)
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


def joint_matches(
    candidates: Sequence[Facet], selections: Mapping[str, FacetSelection]
) -> list[Facet]:
    """The candidates with a `facet_id` carrying every selected ref at once, in candidate
    order. Independent per-axis witnesses can miss a candidate that matches all axes, so the
    joint-coherence checks (`generation._select_member_facets`,
    `group_facets._check_group_joint_coherence`) and dispatch read this set. Empty when no
    candidate jointly matches."""
    return [
        c
        for c in candidates
        if c.facet_id is not None
        and all(
            selection.admits(facet_axis_value(c, axis)) for axis, selection in selections.items()
        )
    ]


def facets_in_window(
    candidates: Sequence[Facet],
    *,
    selections: Mapping[str, FacetSelection],
    window: RequestedWindow | None,
) -> bool | None:
    """Whether a candidate jointly matching `selections` (`joint_matches`) holds a point in
    `window` (`overlaps_window`), `None` when none jointly matches. Its one caller,
    `group_facets.classify_uncounted`, calls a member the election did not count off cadence
    when this is true and out of the window when it is false."""
    matches = joint_matches(candidates, selections)
    if not matches:
        return None
    return any(overlaps_window(facet, window=window) for facet in matches)


def joint_dispatch_facet_ids(
    candidates: Sequence[Facet], election: FacetElection, *, member: int
) -> tuple[str, ...]:
    """Returns the sorted unique ids of the `joint_matches` of `election.selections` among
    member `member`'s `candidates` that serve the election's cadence
    (`FacetElection.serves`). An empty tuple signals no joint match at that cadence, and a
    `None` cadence, an election with no joint match, dispatches nothing."""
    return tuple(
        sorted(
            {
                c.facet_id
                for c in joint_matches(candidates, election.selections)
                # `joint_matches` keeps only facets with an id; this narrows the type.
                if c.facet_id is not None and election.serves(c.facet_id, member=member)
            }
        )
    )


def continuing_facet_ids(
    candidates: Sequence[Facet], election: FacetElection, *, member: int
) -> tuple[str, ...]:
    """The dispatch IDs (`joint_dispatch_facet_ids`) of the method that continues the
    selected series past its last year (`FacetSelection.continued_by`), none when nothing
    continues it. A range window alone continues a series, and its rows serve only the
    years after that series' last one (`columnar.continuation_mask`)."""
    method = election.selections.get("measurementMethod")
    if method is None or method.continued_by is None:
        return ()
    by_id = {c.facet_id: c for c in candidates}
    return tuple(
        facet_id
        for facet_id in joint_dispatch_facet_ids(candidates, election, member=member)
        if by_id[facet_id].measurement_method == method.continued_by
    )


@dataclass(frozen=True)
class Cadence:
    """The one observation period a dispatch serves (`cadences_by_member`): a stated or
    inferred `observationPeriod` in `period`, or, for a facet id stating none without
    bare-year evidence, that id in `facet_id`, so two such facets never share a cadence."""

    period: str | None = None
    facet_id: str | None = None


@dataclass(frozen=True)
class FacetElection:
    """An election's selections, axis to `FacetSelection`, the cadence its dispatch serves
    (`dispatch_cadence`), `None` when no candidate carries every selected ref, and each
    member's cadence of every facet id it holds (`cadences_by_member`), in request order: one
    map for a member's own election, one per member for a group's. Member selection
    (`generation._select_member_facets`) and group selection
    (`group_facets.select_group_facets`) return it, so every dispatch, preview and currency
    signal reads one decision. `members` holds the indexes of the members the cadence was
    counted for (`dispatch_cadence`), each holding a facet at it with a point in a range
    window when any member's facet has one: a group member outside it serves nothing."""

    selections: dict[str, FacetSelection]
    cadence: Cadence | None
    cadence_of: tuple[Mapping[str, Cadence], ...]
    members: frozenset[int] = frozenset()

    def serves(self, facet_id: str, *, member: int) -> bool:
        """Whether member `member`'s facet `facet_id` dispatches at the election's cadence."""
        return at_cadence(facet_id, cadence=self.cadence, cadence_of=self.cadence_of[member])


def at_cadence(
    facet_id: str, *, cadence: Cadence | None, cadence_of: Mapping[str, Cadence]
) -> bool:
    """Whether the facet `facet_id` is at `cadence` (`cadence_of`, one member's map from
    `cadences_by_member`). A `None` cadence, a decision with no joint match, holds no facet."""
    return cadence is not None and cadence_of.get(facet_id) == cadence


def cadences_by_member(members: Sequence[Iterable[Facet]]) -> tuple[dict[str, Cadence], ...]:
    """Each member's facet ids mapped to their cadence, one map per member of `members`, each
    member's candidates as its dispatch reads them.

    A stated `observationPeriod` is the id's own, so every member's copy shares it (the
    earliest in sort order should copies state different ones). An id stating none is `P1Y`
    for a member when every copy that member holds carries annual evidence
    (`Facet.reads_annual`), and otherwise the id's own cadence. Members are judged apart,
    since one unstated id can date one variable by year and another by month: the first
    member's copy is `P1Y` and the second's is not, so no cadence holds both. Each place's
    facet judges its own dates as it is read, and a merge across places keeps the evidence
    only when every merged facet has it, so one place dating the id by month makes it the
    id's own cadence for that member, however the merged bounds read. Members whose copies
    carry no annual evidence share the id's cadence, and the mint refuses their rows should
    they hold two grains (`errors.MixedGrainError`)."""
    held = [list(facets) for facets in members]
    stated: dict[str, str] = {}
    for facet in (facet for facets in held for facet in facets):
        if facet.facet_id is not None and facet.observation_period is not None:
            period = stated.get(facet.facet_id)
            if period is None or facet.observation_period < period:
                stated[facet.facet_id] = facet.observation_period
    cadences: list[dict[str, Cadence]] = []
    for facets in held:
        copies: dict[str, list[Facet]] = {}
        for facet in facets:
            if facet.facet_id is not None:
                copies.setdefault(facet.facet_id, []).append(facet)
        cadences.append(
            {
                facet_id: Cadence(period=stated[facet_id])
                if facet_id in stated
                else Cadence(period="P1Y")
                if all(c.reads_annual for c in copies_of_id)
                else Cadence(facet_id=facet_id)
                for facet_id, copies_of_id in copies.items()
            }
        )
    return tuple(cadences)


def dispatch_cadence(
    matching: Sequence[Facet],
    *,
    member_matches: Sequence[Sequence[Facet]],
    cadence_of: Sequence[Mapping[str, Cadence]],
    stat_vars: Iterable[str],
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None,
    variable_breadth: VariableBreadth = None,
    national_currency: bool = False,
) -> tuple[Facet, Cadence, frozenset[int]] | None:
    """The cadence the election dispatches, the facet of `matching` ranked first among those
    a counted member holds at it, and the members counted for it, or `None` when no member
    holds a candidate carrying every selected ref.

    `member_matches` holds each member's candidates carrying every selected ref
    (`joint_matches`), `cadence_of` each member's `cadences_by_member` map, and `matching`
    those candidates merged across members, which the election's ranking (`rank_pool` under
    `stat_vars`, `window`, `breadth` and `variable_breadth`) orders. A member counts toward a cadence through its
    own copies of the facets the ranking keeps, and, when any member's copy holds a point in
    a range window (`overlaps_window`), only through such copies, since a merged facet's
    dates are every member's together. The cadence held by the most members wins, so a
    group serves every member it can at one cadence. Among cadences held by as many,
    the one whose counted copy ranks first wins, then the one held by the earliest member. The
    facet axes leave out `observationPeriod`, so the UN graph's annual (`P1Y`) and quarterly
    (`P3M`) ILO facets match alike, and dispatching both puts `2019` and `2019-01` into one
    series.

    For one member every cadence it holds counts once, so its cadence is its top facet's: an
    unbound election's elected facet's, and for a bound axis or a graph-declared default the
    facet the ranking, breadth first, puts first among those it admits, so a bound request
    over several places keeps the facet covering the most.

    `national_currency` ranks as `rank_candidates` does under a national-currency unit
    binding."""
    if not matching:
        return None
    pool = rank_pool(
        matching,
        stat_vars=stat_vars,
        window=window,
        breadth=breadth,
        variable_breadth=variable_breadth,
        national_currency=national_currency,
    )
    ranked = pool.ranked
    kept = {facet.facet_id for facet in ranked}
    held = [[f for f in facets if f.facet_id in kept] for facets in member_matches]
    if any(overlaps_window(f, window=window) for facets in held for f in facets):
        held = [[f for f in facets if overlaps_window(f, window=window)] for facets in held]
    members_at: dict[Cadence, list[int]] = {}
    cadences_by_facet_id: dict[str, set[Cadence]] = {}
    for member, (facets, cadences) in enumerate(zip(held, cadence_of, strict=True)):
        seen_cadences: set[Cadence] = set()
        for facet in facets:
            if facet.facet_id is None:
                continue
            cadence = cadences.get(facet.facet_id)
            if cadence is None:
                continue
            cadences_by_facet_id.setdefault(facet.facet_id, set()).add(cadence)
            if cadence not in seen_cadences:
                members_at.setdefault(cadence, []).append(member)
                seen_cadences.add(cadence)
    if not members_at:
        return None
    if isinstance(variable_breadth, PendingBreadth | BreadthUnread) and (
        decided := _cadence_breadth_decides(
            ranked,
            members_at=members_at,
            cadences_by_facet_id=cadences_by_facet_id,
            leading_keys=partial(
                _leading_keys, window=window, precedence=pool.precedence, breadth=breadth
            ),
        )
    ):
        read = _read_breadth(variable_breadth)
        if isinstance(read, Mapping):
            ranked = rank_candidates(
                pool.pool,
                window=window,
                precedence=pool.precedence,
                breadth=breadth,
                variable_breadth=read,
                national_currency=national_currency,
            )
        else:
            _act_on_unread(read, decided)
    rank_of: dict[Cadence, int] = {}
    top_by_cadence: dict[Cadence, Facet] = {}
    for position, facet in enumerate(ranked):
        for cadence in cadences_by_facet_id.get(facet.facet_id, ()):
            rank_of.setdefault(cadence, position)
            top_by_cadence.setdefault(cadence, facet)
    cadence = min(
        members_at,
        key=lambda one: (-len(members_at[one]), rank_of.get(one, len(ranked)), members_at[one][0]),
    )
    return top_by_cadence[cadence], cadence, frozenset(members_at[cadence])


def _cadence_breadth_decides(
    ranked: Sequence[Facet],
    *,
    members_at: Mapping[Cadence, Sequence[int]],
    cadences_by_facet_id: Mapping[str, set[Cadence]],
    leading_keys: Callable[[Facet], tuple[object, ...]],
) -> list[str]:
    """The facet ids whose order under whole-variable breadth would decide `dispatch_cadence`'s
    cadence, empty when no order of them could change it.

    The cadence held by the most members wins first, whatever the order. Among those, the one
    held by the first-ranked facet holding any of them wins, then the one whose first member is
    earliest. Breadth can only reorder facets that tie on every earlier key (`leading_keys`),
    and can put any of them first. So the cadence is decided by breadth exactly when, in the
    first run of tied facets holding a contending cadence, two of them would elect different
    cadences."""
    most = max(len(members) for members in members_at.values())
    contenders = [cadence for cadence, members in members_at.items() if len(members) == most]
    for _, run in groupby(ranked, key=leading_keys):
        holders = [
            (facet, held)
            for facet in run
            if (
                held := [
                    cadence
                    for cadence in contenders
                    if cadence in cadences_by_facet_id.get(facet.facet_id or "", ())
                ]
            )
        ]
        if holders:
            elected = {min(held, key=lambda one: members_at[one][0]) for _, held in holders}
            return [facet.facet_id or "" for facet, _ in holders] if len(elected) > 1 else []
    return []


def facet_election(
    selections: Mapping[str, FacetSelection],
    *,
    matching: Sequence[Facet],
    member_matches: Sequence[Sequence[Facet]],
    cadence_of: Sequence[Mapping[str, Cadence]],
    stat_vars: Iterable[str],
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None,
    variable_breadth: VariableBreadth = None,
) -> FacetElection:
    """The election over `selections`: the cadence `dispatch_cadence` decides, and each
    selection's `dispatch_facet_id` re-pointed to the ranking's top facet at that cadence
    when it names no facet of `matching` at it. A bound axis's or a declared default's
    witness is only the first candidate carrying its value (`select_bound_facet`,
    `_match_policy`), which can sit at another cadence or carry another axis's value apart;
    the top carries every selected ref at the cadence, so the trace cites a facet the
    dispatch serves whatever the candidate order."""
    cadence_of = tuple(cadence_of)
    unit = selections.get("unit")
    decided = dispatch_cadence(
        matching,
        member_matches=member_matches,
        cadence_of=cadence_of,
        stat_vars=stat_vars,
        window=window,
        breadth=breadth,
        variable_breadth=variable_breadth,
        national_currency=unit is not None and unit.national_currency is not None,
    )
    if decided is None:
        return FacetElection(selections=dict(selections), cadence=None, cadence_of=cadence_of)
    top, cadence, members = decided
    assert top.facet_id is not None  # `dispatch_cadence` returns a facet at a cadence.
    matching_ids = {c.facet_id for c in matching if c.facet_id is not None}
    served = {
        facet_id
        for member_cadences in cadence_of
        for facet_id, member_cadence in member_cadences.items()
        if member_cadence == cadence and facet_id in matching_ids
    }
    return FacetElection(
        selections={
            axis: selection
            if selection.dispatch_facet_id in served
            else replace(selection, dispatch_facet_id=top.facet_id)
            for axis, selection in selections.items()
        },
        cadence=cadence,
        cadence_of=cadence_of,
        members=members,
    )


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
    ranked_values: tuple[RankedValue, ...] = (),
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
        ranked_values=ranked_values,
    )


def ranked_values(
    ranked_pool: RankedPool,
    axis: str,
    *,
    reason: str,
    window: RequestedWindow | None,
    breadth: Mapping[str, int] | None,
) -> tuple[RankedValue, ...]:
    """Each distinct value on `axis` among the candidates `ranked_pool` compares, in rank
    order, with the figures the ranking read for the first one carrying it, when `reason`
    is `most_observations_over_disagreement`. Empty for any other reason, where the policy
    chose over no disagreement.

    A joined candidate (`RankedPool.groups`, as the ranking formed it) reports its joined
    facet's years and count and its places across all its counted facets
    (`_joined_counts`), under its lead's value; the estimated facets continuing it rank
    only within it, so carry no value of their own."""
    if reason != "most_observations_over_disagreement":
        return ()
    groups = {id(group.lead): group for group in ranked_pool.groups}
    places = _joined_counts(ranked_pool.groups, breadth)
    by_value: dict[str | None, RankedValue] = {}
    for facet in ranked_pool.ranked:
        group = groups.get(id(facet))
        value = facet_axis_value(facet, axis)
        if group is None or value in by_value:
            continue
        by_value[value] = RankedValue(
            ref=value,
            places=None if places is None else places.get(facet.facet_id or "", 0),
            years=(
                years_in_window(group.joined, window=window)
                if window is not None and window.kind == "range"
                else None
            ),
            observations=group.joined.obs_count,
        )
    return tuple(by_value.values())


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
