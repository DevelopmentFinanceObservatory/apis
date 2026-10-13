"""Elect and memoise each candidate's bound unit for the also-plausible fan-out.

Every candidate decides through `convertibility.decide_part_unit`, the decision a resubmit
of it runs, over its own read plan's probes. The fan-out can include ``max_candidates - 1``
alternates, so shared decisions avoid repeating probes. ``state.unit_elected_selections`` is
not filled yet when the fan-out runs, so this module keeps its own memo. ``bound_facets``
stays empty, as in the elected-shape path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dc_sdmx import SdmxClient as MaterialiseSdmxClient
from materialise import GraphReader, RequestedWindow
from materialise.errors import (
    DefaultFacetUndefinedError,
    FacetNotAvailableError,
    MaterialisationError,
)
from materialise.facets import (
    BreadthUnread,
    CoverageAdmission,
    ElectedDataset,
    FacetSelection,
    MemberProbe,
    PendingBreadth,
    PlaceAdmission,
)
from materialise.facets import UnitRequest as MaterialiseUnitRequest
from materialise.graph import Facet
from search.engine.region.convertibility import decide_part_unit, unit_request_for_materialise
from search.engine.region.phase_shared import UNIT_CONVERSION_REFUSALS
from search.engine.retrieval.elected_dataset import NO_DATASETS, ElectedDatasets
from search.engine.retrieval.read_plan import PlanYears, plan_window
from search.models import IdentityUnitRequest
from series_coverage.index import CoverageIndex

# A candidate StatVar refused for one of these reasons definitely cannot serve the unit, so the
# fan-out omits it and keeps the refusal for the no-survivor answer. Every other
# `MaterialisationError` (an upstream version change, a ceiling that stops the member count,
# members the deployment does not hold, a dataset disagreement) propagates, as it does on the
# elected shape's `decide_part_unit`. The caller drops an alternate whose election raises a
# dataset disagreement (`candidate_grounding._prepare_candidate_context`).
_DEFINITE_UNIT_REFUSALS = (
    FacetNotAvailableError,
    DefaultFacetUndefinedError,
    *UNIT_CONVERSION_REFUSALS,
)

# Key by what one StatVar's election reads: the StatVar, its role/DCID bindings, declared
# roles, the window its years read (`plan_window`), unit request, elected dataset, and its
# probe with the probe's series counts under that window (`_probe_key`). The value is the
# election or the refusal that ended it. Role/DCID pairs distinguish candidates that share
# entities on different axes; the probe distinguishes candidates whose read plans give the
# StatVar different facets.
type _ProbeKey = tuple[
    tuple[Facet, ...],
    frozenset[tuple[str, frozenset[str]]] | CoverageAdmission,
    tuple[str, ...],
    bool,
    frozenset[tuple[str, int]] | None,
    frozenset[tuple[str, int]] | BreadthUnread | PendingBreadth | None,
]
_CandidateUnitElectionKey = tuple[
    str,
    frozenset[tuple[str, tuple[str, ...]]],
    tuple[str, ...],
    RequestedWindow,
    MaterialiseUnitRequest,
    ElectedDataset | None,
    _ProbeKey | None,
]
_CandidateUnitElectionMemo = dict[
    _CandidateUnitElectionKey, Mapping[str, FacetSelection] | MaterialisationError
]


@dataclass(frozen=True)
class CandidateUnitElections:
    """One candidate's elections and refusals, each keyed by StatVar in election order."""

    elected: dict[str, tuple[Mapping[str, FacetSelection], ...]]
    refusals: dict[str, MaterialisationError]


def _candidate_unit_elections(
    sv_dcids: Sequence[str],
    *,
    unit_request: IdentityUnitRequest,
    role_bindings: Mapping[str, Sequence[str]],
    declared_roles: tuple[str, ...],
    probes: Mapping[str, MemberProbe] | None,
    years: PlanYears,
    graph: GraphReader,
    sdmx: MaterialiseSdmxClient,
    coverage: CoverageIndex,
    memo: _CandidateUnitElectionMemo,
    deadline: float | None,
    datasets: ElectedDatasets = NO_DATASETS,
) -> CandidateUnitElections:
    """Elect a unit per StatVar: one selection tuple for each success, and the refusal of
    each StatVar the election proved cannot serve the unit.

    Empty ``role_bindings`` elects nothing. Memo hits skip the decision. Each StatVar
    decides on its own through ``decide_part_unit`` over ``probes``, the candidate's read
    plan probes, within its elected dataset in ``datasets``. A definite refusal
    (``_DEFINITE_UNIT_REFUSALS``) leaves that StatVar out of ``elected``, records it in
    ``refusals`` and memoises it. Any other ``MaterialisationError`` propagates and memoises
    nothing. The caller passes the request's single coverage index unchanged to every
    decision.
    """
    elected: dict[str, tuple[Mapping[str, FacetSelection], ...]] = {}
    refusals: dict[str, MaterialisationError] = {}
    if not role_bindings:
        return CandidateUnitElections(elected=elected, refusals=refusals)
    entity_key = frozenset((role, tuple(dcids)) for role, dcids in role_bindings.items())
    materialise_unit_request = unit_request_for_materialise(unit_request)
    window = plan_window(years)
    for sv in sv_dcids:
        key: _CandidateUnitElectionKey = (
            sv,
            entity_key,
            declared_roles,
            window,
            materialise_unit_request,
            datasets.get(sv),
            _probe_key(probes.get(sv) if probes is not None else None, window),
        )
        if key in memo:
            cached = memo[key]
            if isinstance(cached, MaterialisationError):
                refusals[sv] = cached
            else:
                elected[sv] = (cached,)
            continue
        try:
            (selections,) = decide_part_unit(
                [sv],
                unit_request=unit_request,
                role_bindings=role_bindings,
                declared_roles=declared_roles,
                years=years,
                graph=graph,
                sdmx=sdmx,
                coverage=coverage,
                probes=probes,
                datasets=datasets,
                deadline=deadline,
            )[sv]
        except _DEFINITE_UNIT_REFUSALS as exc:
            memo[key] = exc
            refusals[sv] = exc
            continue
        memo[key] = selections
        elected[sv] = (selections,)
    return CandidateUnitElections(elected=elected, refusals=refusals)


def _probe_key(probe: MemberProbe | None, window: RequestedWindow) -> _ProbeKey | None:
    """`probe` as the election reads it, hashable: its candidates, admission, places, series
    counts under `window`, and places across the whole variable, or the `PendingBreadth` that
    reads them, which keys without reading and is one object per variable per request."""
    if probe is None:
        return None
    admission = probe.admission
    breadth = probe.breadth_in(window)
    return (
        probe.candidates,
        frozenset(admission.places_by_facet.items())
        if isinstance(admission, PlaceAdmission)
        else admission,
        probe.places,
        probe.every_place,
        frozenset(breadth.items()) if breadth is not None else None,
        probe.variable_breadth
        if probe.variable_breadth is None
        or isinstance(probe.variable_breadth, BreadthUnread | PendingBreadth)
        else frozenset(probe.variable_breadth.items()),
    )
