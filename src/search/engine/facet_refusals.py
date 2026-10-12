"""Check whether posted facet bindings are available on every resolved StatVar.

Each binding member needs a facet for every StatVar, pooling observations only
across entities within that StatVar. The window depends on the candidate source:
`facet_refusal` reads entity-keyed facets unwindowed, so a facet with
`obs_count > 0` counts even when its observations fall outside the requested window,
while a discovered role scope's role index uses its request-window rows. Bound role
candidate coverage validates provenance against its lifetime rows and method or unit
against the requested-window rows. A bound role-bearing provenance pin is checked
before a scoped `no_observations` result; other no-data gates precede
`facet_not_available`.

`candidate_facet_refusal` reads the identity facets, `facets.provenance` included.
Provenance uses `region.facets._facet_source_dcid`.
`pin_admitted` runs the same check for the facet options a spec offers.
For request-form units, `elected_units` admits only each StatVar's elected
unit refs. Without that map, `unit_refs.satisfies` tests the request directly, within
the caller's absolute monotonic `deadline`; an expired deadline raises.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, get_args

from dfo_platform import Alternative
from dfo_platform.named import named_capped
from search.engine.clients.graph import EngineGraphClient, Facet
from search.engine.clients.sdmx import RoleFacetIndex
from search.engine.config import settings
from search.engine.errors import EngineInputError
from search.engine.region.facet_probe import NO_PINS, FacetPins, facet_pins
from search.engine.region.facets import _facet_source_dcid, binding_admits_value
from search.engine.region.refs import provenance_labels
from search.engine.retrieval.elected_dataset import NO_DATASETS, ElectedDatasets, published
from search.engine.unit import refs as unit_refs
from search.models import (
    FacetAxis,
    IdentityBindingSet,
    IdentityBindingValue,
    IdentityFacetBinding,
    IdentityFacetValueAbsent,
    IdentityFacetValueRequest,
    IdentityUnitRequest,
)
from search.observations import facet_value_label

RELEVANT_FACET_AXES = ("provenance", "measurementMethod", "unit")
"""facet_refusal judges every FacetAxis member in ResubmitIdentityPayload.identity.facets.
Public: engine.options.surface reads this too, to decide whether a resolved
identity's facet bindings constrain observationAbout option membership at
all (see facets_satisfy_bindings below) and, separately, whether a given
facet counts toward an admitted option's coverage total (see
facet_satisfies_bindings below)."""


def _axis_value(axis: str, facet: Facet) -> str | None:
    if axis == "provenance":
        return _facet_source_dcid(facet)
    return facet.measurement_method if axis == "measurementMethod" else facet.unit


def _unit_request_admits(
    graph: EngineGraphClient,
    request: IdentityUnitRequest,
    facet: Facet,
    sv: str | None,
    elected_units: Mapping[str, frozenset[str]] | None,
    *,
    deadline: float | None,
) -> bool:
    """Whether `facet` honours a request-form unit binding for stat_var
    `sv`. With `elected_units` given, the presence check matches the facet
    a mint would elect: `facet.unit` must be one of `elected_units[sv]` (an
    `sv` missing from the map elects nothing, so it admits no facet). With
    `elected_units` None, the literal rule from before this carve-out
    holds: `unit_refs.satisfies` against the request itself."""
    if facet.unit is None:
        return False
    if elected_units is not None:
        return facet.unit in elected_units.get(sv, frozenset())
    return unit_refs.satisfies(graph, request, facet.unit, deadline=deadline)


def _binding_admits(
    graph: EngineGraphClient,
    axis: str,
    binding: IdentityFacetBinding,
    facet: Facet,
    *,
    sv: str | None = None,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> bool:
    """Whether `facet` satisfies one axis's posted binding -- ref, set and
    null ref by `binding_admits_value`, a unit request form by
    `_unit_request_admits` (`sv`/`elected_units` matter only for that
    form). Mirrors region/resubmit_request.py's
    _facet_filter_from_bindings' per-axis predicates, at per-call rather
    than per-axis-closure granularity, since this module also needs to
    exclude one axis's own binding when checking its members (see
    _other_axes_satisfied)."""
    if isinstance(binding, IdentityBindingValue | IdentityBindingSet | IdentityFacetValueAbsent):
        return binding_admits_value(binding, _axis_value(axis, facet))
    if isinstance(binding, IdentityFacetValueRequest):
        # ResubmitIdentityPayload's own model validator already refuses the
        # request form anywhere but the unit axis, so axis == "unit" here
        # always -- asserted rather than silently testing facet.unit under
        # whichever axis this call happened to be on.
        assert axis == "unit", f"request form reached a non-unit axis: {axis!r}"
        return _unit_request_admits(
            graph, binding.request, facet, sv, elected_units, deadline=deadline
        )
    return True  # IdentityBindingUnbound never reaches this module (see facet_refusal)


def _other_axes_satisfied(
    graph: EngineGraphClient,
    facet: Facet,
    relevant: Mapping[str, IdentityFacetBinding],
    exclude_axis: str,
    *,
    sv: str | None = None,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> bool:
    """Whether `facet` survives every relevant axis's binding OTHER than
    `exclude_axis` -- the all-or-nothing check's "under all the OTHER
    bindings" scoping. `exclude_axis`'s own binding is never consulted
    here, so a member can never be judged missing by its own filter."""
    return all(
        axis == exclude_axis
        or _binding_admits(
            graph, axis, binding, facet, sv=sv, elected_units=elected_units, deadline=deadline
        )
        for axis, binding in relevant.items()
    )


def facet_satisfies_bindings(
    graph: EngineGraphClient,
    facet: Facet,
    bindings: Mapping[str, IdentityFacetBinding],
    *,
    sv: str | None = None,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> bool:
    """Whether `facet` ALONE (unwindowed presence: `obs_count > 0`) honours
    every relevant axis binding in `bindings`, under an ANY-MEMBER reading: a
    set binding is satisfied by this one facet as soon as its axis value is
    ONE of the set's members (`_binding_admits`), not every one of them.
    `bindings` entries outside RELEVANT_FACET_AXES impose no constraint; an
    empty/absent `bindings` is satisfied by any facet with `obs_count > 0`.
    `sv`/`elected_units` are the request-form unit binding's own admission
    rule (see `_unit_request_admits`); a caller not naming `sv` never
    matches an `elected_units` entry, so a bound unit request then admits
    no facet.

    Used by engine.options.surface to aggregate PER-OPTION COVERAGE across an
    already-admitted candidate's facets: once membership
    (`facets_satisfy_bindings` below) has confirmed every set member is
    covered somewhere, a facet counts toward that candidate's coverage total
    as soon as it carries any one of them, regardless of which -- a coverage
    COUNT, not a membership TEST. This is deliberately looser than the
    all-or-nothing rule `facet_refusal`'s own resubmit-time refusal and
    `facets_satisfy_bindings` both apply; it is not the shared reading for
    membership."""
    if facet.obs_count <= 0:
        return False
    return all(
        _binding_admits(
            graph, axis, binding, facet, sv=sv, elected_units=elected_units, deadline=deadline
        )
        for axis, binding in bindings.items()
        if axis in RELEVANT_FACET_AXES
    )


def _member_present(
    graph: EngineGraphClient,
    axis: str,
    member: IdentityBindingValue | IdentityFacetValueAbsent,
    facets_by_sv: Mapping[str, list[Facet]],
    relevant: Mapping[str, IdentityFacetBinding],
    *,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> bool:
    """Whether every sv in `facets_by_sv` has, among ITS OWN facets (pooled
    only across obs_entities, never across stat_vars), at least one that
    `binding_admits_value` admits for `member` (a set binding is checked one
    member at a time) -- the per-stat_var requirement the module docstring
    describes.
    `sv` (the real key from `facets_by_sv`) threads into `_other_axes_satisfied`
    so a bound unit request judged as one of the OTHER axes reads
    `elected_units` under the right stat_var."""
    return all(
        any(
            facet.obs_count > 0
            and binding_admits_value(member, _axis_value(axis, facet))
            and _other_axes_satisfied(
                graph, facet, relevant, axis, sv=sv, elected_units=elected_units, deadline=deadline
            )
            for facet in sv_facets
        )
        for sv, sv_facets in facets_by_sv.items()
    )


def pin_admitted(
    graph: EngineGraphClient,
    axis: str,
    value: str | None,
    facets_by_sv: Mapping[str, list[Facet]],
    pins: FacetPins = NO_PINS,
    *,
    deadline: float | None,
) -> bool:
    """Whether pinning `axis` to `value` (a ref, or the null ref for `None`) together with
    `pins`' other bindings passes this module's availability check over `facets_by_sv`.
    The value replaces any pin `axis` already holds. `_failing_axes` is the check
    `_facet_refusal_from_candidates` refuses on, so over the same candidates that refusal
    rejects the combined pin exactly when this is false and any candidate exists.
    `option_coverage._facet_options` offers a value only when this holds."""
    candidate = (
        IdentityFacetValueAbsent(ref=None) if value is None else IdentityBindingValue(ref=value)
    )
    relevant = {**_relevant(pins.bindings), axis: candidate}
    failing = _failing_axes(
        graph, facets_by_sv, relevant, elected_units=pins.elected_units, deadline=deadline
    )
    return not failing.axes


def pins_admit_facet(
    graph: EngineGraphClient,
    facet: Facet,
    pins: FacetPins,
    *,
    except_axis: str,
    sv: str,
    deadline: float | None,
) -> bool:
    """Whether `facet`, a candidate of StatVar `sv`, honours every pin in `pins` other than
    `except_axis`'s, so a pin on `except_axis` would serve it."""
    return _other_axes_satisfied(
        graph,
        facet,
        _relevant(pins.bindings),
        except_axis,
        sv=sv,
        elected_units=pins.elected_units,
        deadline=deadline,
    )


def _relevant(
    bindings: Mapping[str, IdentityFacetBinding] | None,
) -> dict[str, IdentityFacetBinding]:
    return {
        axis: binding for axis, binding in (bindings or {}).items() if axis in RELEVANT_FACET_AXES
    }


def _request_satisfied(
    graph: EngineGraphClient,
    axis: str,
    request: IdentityUnitRequest,
    facets_by_sv: Mapping[str, list[Facet]],
    relevant: Mapping[str, IdentityFacetBinding],
    *,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> bool:
    """The request-form counterpart of `_member_present`: every sv must have
    its own facet honouring `request` (`_unit_request_admits`, keyed on the
    real `sv` from `facets_by_sv` when `elected_units` is given)."""
    return all(
        any(
            facet.obs_count > 0
            and _unit_request_admits(graph, request, facet, sv, elected_units, deadline=deadline)
            and _other_axes_satisfied(
                graph, facet, relevant, axis, sv=sv, elected_units=elected_units, deadline=deadline
            )
            for facet in sv_facets
        )
        for sv, sv_facets in facets_by_sv.items()
    )


def facets_satisfy_bindings(
    graph: EngineGraphClient,
    facets: list[Facet],
    bindings: Mapping[str, IdentityFacetBinding],
    *,
    sv: str | None = None,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> bool:
    """Whether ONE stat_var's own facet list (pooled only across
    obs_entities, never across stat_vars) honours every relevant axis
    binding in `bindings`, under the SAME all-or-nothing rule
    `facet_refusal`'s own resubmit-time check applies: a set binding is
    satisfied only when EACH of its members has its own facet in THIS list
    (`_member_present`, reused here by wrapping `facets` as a single-entry
    facets_by_sv keyed on `sv` so its per-member/per-sv loop degenerates to a
    per-member check scoped to this one list); a ref binding needs its one
    member the same way; a unit request binding needs `_request_satisfied`.
    `bindings` entries outside RELEVANT_FACET_AXES impose no constraint.
    `sv` names the stat_var `facets` belongs to, for `elected_units`
    lookups on a bound unit request; a caller not naming `sv` never matches
    an `elected_units` entry.

    Used by engine.options.surface to decide observationAbout option
    membership one stat_var at a time -- a candidate is a member only when
    every resolved stat_var's own facet list passes this check, mirroring
    `facet_refusal`'s all-or-nothing set semantics per stat_var rather than
    `facet_satisfies_bindings`' looser any-member, single-facet reading."""
    relevant = {axis: binding for axis, binding in bindings.items() if axis in RELEVANT_FACET_AXES}
    facets_by_sv = {sv if sv is not None else "_": facets}
    for axis, binding in relevant.items():
        if isinstance(binding, IdentityBindingSet):
            if not all(
                _member_present(
                    graph,
                    axis,
                    IdentityBindingValue(ref=member),
                    facets_by_sv,
                    relevant,
                    elected_units=elected_units,
                    deadline=deadline,
                )
                for member in binding.refs
            ):
                return False
        elif isinstance(binding, IdentityBindingValue | IdentityFacetValueAbsent):
            if not _member_present(
                graph,
                axis,
                binding,
                facets_by_sv,
                relevant,
                elected_units=elected_units,
                deadline=deadline,
            ):
                return False
        elif isinstance(binding, IdentityFacetValueRequest) and not _request_satisfied(
            graph,
            axis,
            binding.request,
            facets_by_sv,
            relevant,
            elected_units=elected_units,
            deadline=deadline,
        ):
            return False
    return True


def _alternatives(
    axis: str, all_facets: list[Facet], graph: EngineGraphClient
) -> list[Alternative]:
    """Every distinct value `axis` DOES carry (unwindowed presence),
    descending by summed observation count then ascending by dcid -- mirrors
    region/option_coverage.py's _facet_options ranking. A provenance the origin
    catalogue lists takes its label. Any other dcid the graph can't label falls
    back to the dcid itself, same convention as _facet_options. Capped at
    settings().max_options."""
    counts: dict[str, int] = {}
    for facet in all_facets:
        value = _axis_value(axis, facet)
        if value is not None and facet.obs_count > 0:
            counts[value] = counts.get(value, 0) + facet.obs_count
    if not counts:
        return []
    ranked = sorted(counts, key=lambda dcid: (-counts[dcid], dcid))[: settings().max_options]
    labels = (
        provenance_labels(ranked, graph)
        if axis == "provenance"
        else graph.node_labels_batch(ranked)
    )
    return [
        Alternative(ref=dcid, label=facet_value_label(axis, labels.get(dcid, dcid)))
        for dcid in ranked
    ]


class _FailingAxes(NamedTuple):
    """The axes whose binding a candidate set fails, in `FacetAxis` order, with the missing
    ref or set members per axis and the failed request per request-form axis."""

    axes: list[str]
    missing: dict[str, list[str]]
    requests: dict[str, IdentityUnitRequest]


def _failing_axes(
    graph: EngineGraphClient,
    facets_by_sv: Mapping[str, list[Facet]],
    relevant: Mapping[str, IdentityFacetBinding],
    *,
    provenance_facets_by_sv: Mapping[str, list[Facet]] | None = None,
    elected_units: Mapping[str, frozenset[str]] | None,
    deadline: float | None,
) -> _FailingAxes:
    """Check every relevant binding under the per-StatVar all-or-nothing rule.

    ``provenance_facets_by_sv`` holds a source pin's lifetime candidates, while
    ``facets_by_sv`` holds the requested-window candidates for method and unit.
    Each axis still checks every other binding against a facet from its own
    view, retaining joint-pin validation.
    """
    failing = _FailingAxes([], {}, {})
    for axis in get_args(FacetAxis):
        binding = relevant.get(axis)
        if binding is None:
            continue
        candidates = (
            provenance_facets_by_sv
            if axis == "provenance" and provenance_facets_by_sv is not None
            else facets_by_sv
        )
        if isinstance(binding, IdentityBindingSet):
            missing = [
                member
                for member in binding.refs
                if not _member_present(
                    graph,
                    axis,
                    IdentityBindingValue(ref=member),
                    candidates,
                    relevant,
                    elected_units=elected_units,
                    deadline=deadline,
                )
            ]
            if missing:
                failing.axes.append(axis)
                failing.missing[axis] = missing
        elif isinstance(binding, IdentityBindingValue | IdentityFacetValueAbsent):
            if not _member_present(
                graph,
                axis,
                binding,
                candidates,
                relevant,
                elected_units=elected_units,
                deadline=deadline,
            ):
                failing.axes.append(axis)
                failing.missing[axis] = [
                    binding.ref if binding.ref is not None else f"a facet with no {axis}"
                ]
        elif isinstance(binding, IdentityFacetValueRequest):
            if not _request_satisfied(
                graph,
                axis,
                binding.request,
                candidates,
                relevant,
                elected_units=elected_units,
                deadline=deadline,
            ):
                failing.axes.append(axis)
                failing.requests[axis] = binding.request
    return failing


def _facet_refusal_from_candidates(
    graph: EngineGraphClient,
    facets_by_sv: Mapping[str, list[Facet]],
    facets: Mapping[str, IdentityFacetBinding] | None,
    *,
    provenance_facets_by_sv: Mapping[str, list[Facet]] | None = None,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> EngineInputError | None:
    """Apply facet_refusal's availability contract to already-discovered
    candidate facets, rather than reading a new entity-keyed facet batch.

    The entity path, the discovered role scope and the bound role path all use this
    helper, and `pin_admitted` shares its `_failing_axes` check, so ref/set membership,
    unit requests, and the per-StatVar all-or-nothing rule cannot drift between candidate
    sources or between a refusal and the options it offers.
    """
    relevant = _relevant(facets)
    if not relevant:
        return None

    unit_binding = relevant.get("unit")
    if isinstance(unit_binding, IdentityFacetValueRequest) and not unit_refs.expressible(
        unit_binding.request
    ):
        detail = (
            "the requested currency is neither an ISO 4217 code nor a role shape, "
            "so no conversion can express it."
        )
        return EngineInputError(
            detail,
            code="facet_binding_unsupported",
            status=501,
            pointer="/input/facets/unit",
        )

    all_facets = [f for sv_facets in facets_by_sv.values() for f in sv_facets]
    lifetime_by_sv = facets_by_sv if provenance_facets_by_sv is None else provenance_facets_by_sv
    provenance_facets = [f for sv_facets in lifetime_by_sv.values() for f in sv_facets]
    if not provenance_facets:
        return None

    failing = _failing_axes(
        graph,
        facets_by_sv,
        relevant,
        provenance_facets_by_sv=provenance_facets_by_sv,
        elected_units=elected_units,
        deadline=deadline,
    )
    if not failing.axes:
        return None

    pointed_axis = failing.axes[0]
    detail_parts: list[str] = []
    for axis in failing.axes:
        if axis in failing.requests:
            req = failing.requests[axis]
            basis = f"{req.currency}/{req.prices}" + (
                f"/{req.base_year}" if req.base_year is not None else ""
            )
            detail_parts.append(f"{axis}: no facet honours the requested {basis}")
        else:
            detail_parts.append(f"{axis}: missing {named_capped(failing.missing[axis])}")
    detail = "no observations are available under the requested facet binding(s) -- " + "; ".join(
        detail_parts
    )
    return EngineInputError(
        detail,
        code="facet_not_available",
        status=422,
        pointer=f"/input/facets/{pointed_axis}",
        alternatives=_alternatives(
            pointed_axis,
            provenance_facets if pointed_axis == "provenance" else all_facets,
            graph,
        )
        or None,
    )


def candidate_facet_refusal(
    graph: EngineGraphClient,
    facets_by_sv: Mapping[str, list[Facet]],
    facets: Mapping[str, IdentityFacetBinding] | None,
    *,
    provenance_facets_by_sv: Mapping[str, list[Facet]] | None = None,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    deadline: float | None,
) -> EngineInputError | None:
    """Validate posted facet bindings against each StatVar's candidate facets.

    Method and unit use the requested-window candidates. When passed, provenance uses
    lifetime candidates, while every check still joins the other pin axes on one facet.
    """
    pins = facet_pins(facets)
    return _facet_refusal_from_candidates(
        graph,
        facets_by_sv,
        pins.bindings,
        provenance_facets_by_sv=provenance_facets_by_sv,
        elected_units=elected_units,
        deadline=deadline,
    )


def role_index_facets(role_index_by_sv: RoleFacetIndex) -> dict[str, list[Facet]]:
    """A discovered role scope's per-StatVar candidates as facets.

    The coverage-built role index's rows establish that each candidate exists within the
    request's window, though they do not carry an observation count. Each candidate becomes
    one positive synthetic Facet, with obs_count set to its pair count so `_alternatives`
    ranks values by how many pairs carry them. No observations are fetched: a role scope
    has no single subject entity to ask the entity-keyed facet endpoint about.
    """
    return {
        sv: [
            Facet(
                earliest_date=None,
                latest_date=candidate.latest_date,
                obs_count=max(candidate.pair_count, 1),
                provenance_id=candidate.provenance_id,
                measurement_method=candidate.measurement_method,
                unit=candidate.unit,
                scaling_factor=candidate.scaling_factor,
                facet_id=candidate.facet_id,
            )
            for candidate in candidates
        ]
        for sv, candidates in role_index_by_sv.items()
    }


def facet_refusal(
    graph: EngineGraphClient,
    sv_dcids: list[str],
    obs_entities: list[str],
    facets: Mapping[str, IdentityFacetBinding] | None,
    *,
    elected_units: Mapping[str, frozenset[str]] | None = None,
    datasets: ElectedDatasets = NO_DATASETS,
    deadline: float | None,
) -> EngineInputError | None:
    """Return a facet-availability refusal, or `None` when bindings pass.

    A StatVar with an elected dataset in `datasets` is judged on that dataset's facets
    alone, the ones a mint of the Spec can serve.

    An unexpressible request-form unit returns `facet_binding_unsupported`/501
    before graph reads. Otherwise, each ref or set member must have a
    satisfying facet for every StatVar, under the other axis bindings;
    failures return `facet_not_available`/422 with all failing axes and
    members. Request-form units use `elected_units` when supplied, or
    `unit_refs.satisfies` otherwise. Empty inputs or no candidate facets return
    `None`; callers handle `no_observations` first.
    """
    relevant = (facets or {}).keys() & RELEVANT_FACET_AXES
    if not relevant:
        return None
    unit_binding = (facets or {}).get("unit")
    if isinstance(unit_binding, IdentityFacetValueRequest) and not unit_refs.expressible(
        unit_binding.request
    ):
        # The data-independent unsupported-unit response must precede the
        # entity facet read, just as it did before candidates were factored
        # into _facet_refusal_from_candidates.
        return _facet_refusal_from_candidates(
            graph, {}, facets, elected_units=elected_units, deadline=deadline
        )
    if not sv_dcids or not obs_entities:
        return _facet_refusal_from_candidates(
            graph, {}, facets, elected_units=elected_units, deadline=deadline
        )
    facets_batch = graph.observation_facets_batch(sv_dcids, obs_entities, needs_dates=False)
    facets_by_sv: dict[str, list[Facet]] = {
        sv: published(
            datasets,
            sv,
            (f for entity in obs_entities for f in facets_batch.get(sv, {}).get(entity, [])),
        )
        for sv in sv_dcids
    }
    return _facet_refusal_from_candidates(
        graph, facets_by_sv, facets, elected_units=elected_units, deadline=deadline
    )
