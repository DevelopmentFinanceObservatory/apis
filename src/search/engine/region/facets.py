"""Resolve facet metadata and apply a request's facet filter to an answer.

One matching rule (`facet_admitted`, and per StatVar `FacetAdmission`) decides which facets
a filter admits, over facets and role candidates alike. `filter_answer` applies it once,
before any election, and rebuilds coverage, the headline and the preview's read plan from
the admitted facets.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, assert_never, get_args

from code_index.shared import get_index
from dfo_platform.roles import FlowAccount
from materialise.facets import native_currency_iso
from search.engine.clients.graph import Facet
from search.engine.region.equivalent_series import CLUSTER_BY_MEMBER
from search.engine.region.result import FacetAxisFilter
from search.engine.retrieval.coverage import (
    answer_coverage,
    answer_headline,
    citable_source_ids,
)
from search.engine.retrieval.read_plan import facets_served, series_facets
from search.engine.unit.lexicon import CURRENCY_DISPLAY_NAMES
from search.models import (
    Coverage,
    FacetAxis,
    FacetBasis,
    Headline,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityUnitRequest,
    ResolvedFacet,
    SeriesFacetRef,
)
from search.observations import CURRENCY_DCID_PREFIX, currency_unit_ref, series_facet_map

if TYPE_CHECKING:
    from search.engine.extract import DateRequest
    from search.engine.retrieval.read_plan import ReadPlan
    from search.engine.retrieval.retrieve import Materialised
    from series_coverage.index import CoverageIndex


def _facet_source_dcid(facet: Facet) -> str | None:
    """Return the facet's `provenanceId`, or None when it is absent."""
    dcids = citable_source_ids([facet])
    return dcids[0] if dcids else None


def binding_admits_value(
    binding: IdentityBindingValue | IdentityBindingSet | IdentityFacetValueAbsent,
    value: str | None,
) -> bool:
    """Whether a posted facet binding admits a facet whose axis value is `value`
    (`materialise.facet_axis_value`, `None` when the facet declares none).

    A ref admits an equal value and a set any member. A null ref admits a facet that
    declares no value on the axis. Materialise's `select_facet` matches a bound ref by the
    same equality, `None` included, so the facets a posted binding admits are the facets a
    pinned one serves.
    """
    if isinstance(binding, IdentityBindingSet):
        return value in binding.refs
    return value == binding.ref


class _AxisFacetCandidate(Protocol):
    """A facet's provenance, method and unit, read-only. Facets and bulk candidates satisfy
    it structurally."""

    @property
    def provenance_id(self) -> str | None: ...
    @property
    def measurement_method(self) -> str | None: ...
    @property
    def unit(self) -> str | None: ...


def candidate_facet_refs(candidate: _AxisFacetCandidate) -> dict[str, str]:
    """The axis refs `candidate` declares, in `FacetAxis` order, omitting an axis it leaves
    unset.

    `assert_never` makes a new `FacetAxis` value a type error and runtime assertion.
    """
    result: dict[str, str] = {}
    for raw_axis in get_args(FacetAxis):
        axis: FacetAxis = raw_axis
        match axis:
            case "provenance":
                ref = candidate.provenance_id
            case "measurementMethod":
                ref = candidate.measurement_method
            case "unit":
                ref = candidate.unit
            case _:
                assert_never(axis)
        if ref is not None:
            result[axis] = ref
    return result


def _bulk_candidate_facet_map(
    candidate: _AxisFacetCandidate, labels: Mapping[str, str]
) -> dict[str, SeriesFacetRef]:
    """Build the sparse facet map of `candidate`'s axis refs (`candidate_facet_refs`). The
    provenance axis takes its label, source and Data Commons from the origin catalogue, else
    its label from `labels`. Other axes take `labels` (`search.observations.series_facet_map`).
    """
    return series_facet_map(candidate_facet_refs(candidate), labels)


def _resolved_facets(
    series_facet_maps: Sequence[Mapping[str, SeriesFacetRef]],
    bound_axes: Mapping[str, FacetBasis],
    *,
    request_basis: Mapping[str, FacetBasis] | None = None,
    currency_of: Callable[[str], str | None] | None = None,
) -> dict[str, ResolvedFacet]:
    """Aggregate per-series facets into `Spec.resolved.facets`.

    An axis is omitted when no series carries it. A shared ref on every series yields
    `ref` with the first-seen series' label, source, and Data Commons, and on the provenance
    axis the account of a declared dataset (`_declared_account`). Differing refs or a
    missing value on any series yield `selection="per_series"` without a ref or label. `bound_axes` supplies each axis's
    basis. Unbound axes use `fallback`. `request_basis` is copied to the unit axis only, and
    a shared unit takes its ISO 4217 code from `currency_of` (`unit.refs.unit_currency`).
    Keys follow `FacetAxis` order.
    """
    result: dict[str, ResolvedFacet] = {}
    for axis in get_args(FacetAxis):
        distinct: dict[str, SeriesFacetRef] = {}
        n_with_axis = 0
        for facet_map in series_facet_maps:
            ref = facet_map.get(axis)
            if ref is not None:
                n_with_axis += 1
                distinct.setdefault(ref.ref, ref)
        if not distinct:
            continue
        basis = bound_axes.get(axis, "fallback")
        axis_request_basis = dict(request_basis) if axis == "unit" and request_basis else None
        # A carried series lacking this axis entirely (n_with_axis short of
        # every series) forces the heterogeneous per_series branch below,
        # the same way a genuine second distinct ref would.
        if len(distinct) == 1 and n_with_axis == len(series_facet_maps):
            shared = next(iter(distinct.values()))
            result[axis] = ResolvedFacet(
                basis=basis,
                ref=shared.ref,
                label=shared.label,
                source=shared.source,
                data_commons=shared.data_commons,
                documentation=shared.documentation,
                currency=(
                    currency_of(shared.ref) if axis == "unit" and currency_of is not None else None
                ),
                account=_declared_account(shared.ref) if axis == "provenance" else None,
                request_basis=axis_request_basis,
            )
        else:
            result[axis] = ResolvedFacet(
                basis=basis, selection="per_series", request_basis=axis_request_basis
            )
    return result


def _declared_account(ref: str) -> FlowAccount | None:
    """The flow account of the dataset `ref` when the code index declares it
    (`DatasetDeclaration.account`), else `None`."""
    dataset = get_index().datasets.get(ref)
    return dataset.account if dataset is not None else None


def _currency_label(ref: str, labels: Mapping[str, str]) -> str:
    """`ref`'s graph name from `labels`, else its ISO code's display name in the unit lexicon,
    else the code itself, so a failed label read still names the currency and not the dcid
    tail."""
    if (label := labels.get(ref)) is not None:
        return label
    code = ref.removeprefix(CURRENCY_DCID_PREFIX)
    return CURRENCY_DISPLAY_NAMES.get(code, code)


def _requested_unit_ref(unit_request: IdentityUnitRequest) -> str | None:
    """`currency_unit_ref` for an identity unit request; a role object names no ref."""
    currency = unit_request.currency if isinstance(unit_request.currency, str) else None
    return currency_unit_ref(currency)


def answer_facet_dcids(plan: ReadPlan | None, unit_request: IdentityUnitRequest | None) -> set[str]:
    """The method and unit refs `answer_facets` labels: those of the facets `plan` serves,
    and with `unit_request` the currency it converts to. Grounding adds them to its one label
    batch, so `answer_facets` makes no request of its own."""
    dcids = {
        ref
        for facet in series_facets(plan)
        for ref in (facet.measurement_method, facet.unit)
        if ref is not None
    }
    if unit_request is not None and (ref := _requested_unit_ref(unit_request)) is not None:
        dcids.add(ref)
    return dcids


def answer_facets(
    plan: ReadPlan | None,
    labels: Mapping[str, str],
    *,
    bound_axes: Mapping[str, FacetBasis],
    request_basis: Mapping[str, FacetBasis] | None,
    unit_request: IdentityUnitRequest | None,
    currency_of: Callable[[str], str | None],
) -> dict[str, ResolvedFacet]:
    """`Spec.resolved.facets` for an answer: the facets its read plan serves, aggregated
    across its series (`_resolved_facets`). Every answer builds them here at grounding, on a
    first answer and on a resubmit, whether or not its preview is read, so the preview
    setting never changes them.

    `unit_request` is the request a preview converts per elected unit
    (`observations.fetch._fetch_pair_observations`'s elected mode), None otherwise. Its
    series report the currency it converts to (`currency_unit_ref`), which a natively
    served series already carries, and no unit for a role currency. Labels come from
    `labels`, the grounding batch `answer_facet_dcids` fills, and provenance from the origin
    catalogue first (`_bulk_candidate_facet_map`). A converted currency the batch did not
    label takes its display name (`_currency_label`). A unit served by every series takes
    the ISO 4217 code `currency_of` (`unit.refs.unit_currency`) names for it."""
    converted = _requested_unit_ref(unit_request) if unit_request is not None else None
    maps: list[dict[str, SeriesFacetRef]] = []
    for facet in series_facets(plan):
        facet_map = _bulk_candidate_facet_map(facet, labels)
        if unit_request is not None:
            facet_map.pop("unit", None)
            if converted is not None:
                facet_map["unit"] = SeriesFacetRef(
                    ref=converted, label=_currency_label(converted, labels)
                )
        maps.append(facet_map)
    return _resolved_facets(maps, bound_axes, request_basis=request_basis, currency_of=currency_of)


# ---------------------------------------------------------------------------
# A request's facet filter narrows the read plan, and every count, date and scope member in
# a facet-bound response is rebuilt from the narrowed plan, so it reflects the facets its
# observations do.
# ---------------------------------------------------------------------------


def facet_admitted(
    facet_filter: Mapping[str, FacetAxisFilter] | None, facet: _AxisFacetCandidate
) -> bool:
    """The request's facet matching rule: whether every axis predicate of `facet_filter`
    admits `facet`'s provenance, method and unit. Each predicate tests one axis value
    through `resubmit_request._facet_binding_admits`. Coverage, the headline, discovery and
    the preview all narrow their candidates through here before they elect."""
    if not facet_filter:
        return True
    stub = (
        facet
        if isinstance(facet, Facet)
        else Facet(
            earliest_date=None,
            latest_date=None,
            obs_count=0,
            provenance_id=facet.provenance_id,
            measurement_method=facet.measurement_method,
            unit=facet.unit,
        )
    )
    return all(af.predicate(stub) for af in facet_filter.values())


def facet_candidate_filter(
    facet_filter: Mapping[str, FacetAxisFilter] | None,
) -> Callable[[Facet], bool] | None:
    """`facet_admitted` as a materialise `candidate_filter`, or None with no filter."""
    if not facet_filter:
        return None
    return lambda facet: facet_admitted(facet_filter, facet)


def narrow_facets_to_filter(
    facets: list[Facet], facet_filter: Mapping[str, FacetAxisFilter] | None
) -> list[Facet]:
    """Keep facets `facet_admitted` admits. Return the input unchanged with no filter."""
    if not facet_filter:
        return facets
    return [f for f in facets if facet_admitted(facet_filter, f)]


def narrow_facets_by_sv_to_filter(
    facets_by_sv: dict[str, list[Facet]] | None,
    facet_filter: Mapping[str, FacetAxisFilter] | None,
) -> dict[str, list[Facet]] | None:
    """Narrow each StatVar's facets independently so its date range reflects the filter."""
    if facets_by_sv is None or not facet_filter:
        return facets_by_sv
    return {sv: narrow_facets_to_filter(fs, facet_filter) for sv, fs in facets_by_sv.items()}


def cluster_currency_filter(mat: Materialised) -> dict[str, FacetAxisFilter] | None:
    """The unit filter of an answer whose StatVars all belong to one twin cluster naming a
    currency (`equivalent_series.CLUSTER_BY_MEMBER`), when its facets report that currency
    beside others: it keeps the facets in that currency alone. IMF WEO GDP reports national
    currency beside US dollars and is World Bank GDP's twin in US dollars alone, so either
    serves US dollars whatever else is on the menu. A facet stating no unit is left out, as
    the identity binds the currency (`cluster_currency_binding`) and a resubmit of it admits
    only facets reporting that currency, so the answer serves what its resubmit serves.
    None for every other answer. A request that binds a unit decides the facets instead, so
    callers apply this only when none does."""
    clusters = {CLUSTER_BY_MEMBER.get(dcid) for dcid in mat.sv_dcids}
    cluster = clusters.pop() if len(clusters) == 1 else None
    if cluster is None or cluster.currency is None:
        return None
    currency = cluster.currency
    units = {f.unit for facets in (mat.facets_by_sv or {}).values() for f in facets if f.unit}
    if not any(native_currency_iso(unit) == currency for unit in units) or all(
        native_currency_iso(unit) == currency for unit in units
    ):
        return None
    return {
        "unit": FacetAxisFilter(
            predicate=lambda f: f.unit is not None and native_currency_iso(f.unit) == currency
        )
    }


def cluster_currency_binding(facets: Sequence[Facet]) -> IdentityFacetBinding:
    """The identity's unit binding for an answer narrowed to its cluster's currency
    (`cluster_currency_filter`). It holds the unit refs of its remaining facets, so a mint of
    the Spec serves the same facets."""
    refs = sorted({f.unit for f in facets if f.unit is not None})
    return IdentityBindingValue(ref=refs[0]) if len(refs) == 1 else IdentityBindingSet(refs=refs)


@dataclass(frozen=True)
class FacetAdmission:
    """A request's facet filter as one rule over a facet's provenance, method and unit: the
    axis predicates `axes` (`facet_admitted`), and per StatVar the units `elected_units`
    names when a unit request was elected per StatVar."""

    axes: Mapping[str, FacetAxisFilter] | None = None
    elected_units: Mapping[str, frozenset[str]] | None = None

    def __bool__(self) -> bool:
        return bool(self.axes) or self.elected_units is not None

    def admits(self, stat_var: str, facet: _AxisFacetCandidate) -> bool:
        if self.elected_units is not None and facet.unit not in self.elected_units.get(
            stat_var, frozenset()
        ):
            return False
        return facet_admitted(self.axes, facet)


@dataclass(frozen=True)
class FilteredAnswer:
    """An answer under a request's facet filter: the preview's read plan narrowed by the
    filter, and the coverage and headline that plan states. Grounding cites the plan's
    served facets (`cited_facets`)."""

    coverage: Coverage
    headline: Headline | None
    plan: ReadPlan | None


def cited_facets(plan: ReadPlan | None) -> list[Facet]:
    """The facets an answer cites: the ones its read plan serves (`read_plan.facets_served`).

    Grounding names the sentence's source and `trace.resolved_sources` from them
    (`grounding._ground_answer`), and coverage counts its sources from the same list. A
    provenance holding data only outside the years read, or one the plan did not elect, is
    absent from all three, on a first answer and on a resubmit alike. An answer with no
    plan reads no facet and cites none."""
    return facets_served(plan) if plan is not None else []


def filter_answer(
    mat: Materialised,
    admission: FacetAdmission,
    *,
    date_request: DateRequest | None,
    coverage_index: CoverageIndex,
) -> FilteredAnswer:
    """Apply a request's facet filter once, before any election: the read plan narrows to the
    facets `admission` admits (`ReadPlan.narrowed`), and coverage and the headline are built
    from the narrowed plan as resolve built them from the unfiltered one
    (`coverage.answer_coverage`, `coverage.answer_headline`). The preview dispatch reads the
    narrowed plan. With no filter, or an answer that read no facets, the answer is unchanged.
    """
    if not admission or mat.read_plan is None or mat.coverage_form is None:
        return FilteredAnswer(mat.coverage, mat.headline, mat.read_plan)
    return plan_answer(
        mat,
        mat.read_plan.narrowed(admission.admits),
        date_request=date_request,
        coverage_index=coverage_index,
    )


def plan_answer(
    mat: Materialised,
    plan: ReadPlan,
    *,
    date_request: DateRequest | None,
    coverage_index: CoverageIndex,
) -> FilteredAnswer:
    """The answer `plan` serves: its coverage stated in `mat`'s form and its headline, built
    from the plan as resolve built them (`coverage.answer_coverage`,
    `coverage.answer_headline`). A carrier with no coverage form keeps its own coverage and
    headline."""
    if mat.coverage_form is None:
        return FilteredAnswer(mat.coverage, mat.headline, plan)
    return FilteredAnswer(
        coverage=answer_coverage(plan, mat.coverage_form),
        headline=answer_headline(plan, coverage_index, date_request=date_request),
        plan=plan,
    )


class FacetsBySvRequiredError(Exception):
    """Raised when `facets_by_sv` is None. Elected units are keyed by StatVar."""


def narrow_by_elected_unit(
    facets_by_sv: dict[str, list[Facet]] | None,
    elected_units: Mapping[str, frozenset[str]],
) -> dict[str, list[Facet]]:
    """Narrow each StatVar's facets to its own elected units, so one StatVar's elected
    units never apply to another's.

    A StatVar absent from `elected_units`, or present with an empty set, keeps no facet.
    `facets_by_sv=None` raises `FacetsBySvRequiredError` because a flat facet list lacks
    StatVar keys.
    """
    if facets_by_sv is None:
        raise FacetsBySvRequiredError(
            "narrow_by_elected_unit requires facets_by_sv: elected_units is "
            "keyed by stat_var, and a flat facet list carries no stat_var of "
            "its own to narrow by."
        )
    return {
        sv: [f for f in facets if f.unit in elected_units.get(sv, frozenset())]
        for sv, facets in facets_by_sv.items()
    }
