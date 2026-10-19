"""Facet selection for a member of an index-resolved family reads its elected dataset alone.

The family's provenance is pinned from the election before any candidate is compared, so
another dataset's observations on a member neither refuse the group nor win the selection,
and a member the probe finds only under provenances the code index does not record for it
is a fault naming the dataset.
"""

from __future__ import annotations

import dataclasses

import pytest

from materialise.errors import DatasetDisagreementError, FacetNotAvailableError
from materialise.facets import (
    CoverageAdmission,
    ElectedDataset,
    coverage_probe,
    dataset_candidates,
    probe_collection_candidates,
    role_bearing_facet_candidates,
    role_bearing_facet_candidates_by_donor,
    role_bearing_facets,
)
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.group_facets import select_group_facets
from origin_catalogue import BASE, DFO, UN_SYSTEM, bound_catalogue
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index
from tests.origin_catalogue.fake_catalogue import (
    FakeCatalogue,
    provenance_row,
    restrictive_catalogue,
)

from ._group_support import _BARE_A, _ENTITY, _STAT_VAR_A, _STAT_VAR_B, _request
from ._scope_availability import NO_SCOPE_READ
from .conftest import FakeGraphReader

_ELECTED = "provenance/Elected"
_OTHER = "provenance/Other"
_UNRECORDED = "provenance/Unrecorded"


def _facet(provenance: str, *, obs_count: int = 1) -> Facet:
    return Facet(
        facet_id=f"facet-{provenance}",
        provenance_id=provenance,
        measurement_method="METH",
        unit="UNIT",
        obs_count=obs_count,
        earliest_date="2015",
        latest_date="2015",
    )


def _member(stat_var: str, *, recorded: tuple[str, ...] = (_ELECTED,), pinned: bool = True):
    request = _request(stat_var=stat_var, facet_bindings={}, index_sourced=True)
    if not pinned:
        return request
    return dataclasses.replace(
        request, dataset=ElectedDataset(provenance=_ELECTED, member_provenances=recorded)
    )


def test_a_group_whose_members_span_datasets_selects_the_elected_provenance():
    """Member B holds observations at the bound entities under another dataset the index
    records for it, and none under the elected one, so it is unpopulated."""
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B, recorded=(_ELECTED, _OTHER))]
    candidates = [[_facet(_ELECTED)], [_facet(_OTHER, obs_count=9)]]

    selections = select_group_facets(requests, candidates, graph=FakeGraphReader()).selections

    assert selections["provenance"].ref == _ELECTED


def test_the_same_group_without_an_election_refuses_for_mixed_provenance():
    requests = [_member(_STAT_VAR_A, pinned=False), _member(_STAT_VAR_B, pinned=False)]
    candidates = [[_facet(_ELECTED)], [_facet(_OTHER, obs_count=9)]]

    with pytest.raises(FacetNotAvailableError, match="no probed facet serves every member"):
        select_group_facets(requests, candidates, graph=FakeGraphReader())


def test_a_member_selects_the_elected_provenance_over_a_larger_facet_of_another_dataset():
    request = _member(_STAT_VAR_A, recorded=(_ELECTED, _OTHER))
    candidates = [_facet(_OTHER, obs_count=9), _facet(_ELECTED)]

    selections = _select_member_facets(request, candidates, graph=FakeGraphReader()).selections

    assert selections["provenance"].ref == _ELECTED
    assert selections["provenance"].dispatch_facet_id == f"facet-{_ELECTED}"


def test_a_member_ranks_only_the_elected_dataset_however_many_places_another_covers():
    """The election leads with each facet's series count (`rank_candidates`), and it ranks
    only the elected dataset's facets, so another dataset covering more places never wins."""
    request = _member(_STAT_VAR_A, recorded=(_ELECTED, _OTHER))
    candidates = [_facet(_OTHER, obs_count=9), _facet(_ELECTED)]

    selections = _select_member_facets(
        request,
        candidates,
        graph=FakeGraphReader(),
        breadth={f"facet-{_OTHER}": 5, f"facet-{_ELECTED}": 1},
    ).selections

    assert selections["provenance"].ref == _ELECTED
    assert selections["provenance"].selection_reason == "sole_candidate"


def test_a_group_ranks_only_the_elected_dataset_however_many_places_another_covers():
    requests = [
        _member(_STAT_VAR_A, recorded=(_ELECTED, _OTHER)),
        _member(_STAT_VAR_B, recorded=(_ELECTED, _OTHER)),
    ]
    candidates = [
        [_facet(_OTHER, obs_count=9), _facet(_ELECTED)],
        [_facet(_OTHER, obs_count=9), _facet(_ELECTED)],
    ]

    selections = select_group_facets(
        requests,
        candidates,
        graph=FakeGraphReader(),
        breadth={f"facet-{_OTHER}": 10, f"facet-{_ELECTED}": 2},
    ).selections

    assert selections["provenance"].ref == _ELECTED
    assert selections["provenance"].dispatch_facet_id == f"facet-{_ELECTED}"


def test_a_member_found_only_under_an_unrecorded_provenance_is_a_fault_naming_the_dataset():
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B)]
    candidates = [[_facet(_ELECTED)], [_facet(_UNRECORDED)]]

    with pytest.raises(DatasetDisagreementError) as excinfo:
        select_group_facets(requests, candidates, graph=FakeGraphReader())

    assert excinfo.value.dataset == _ELECTED
    assert excinfo.value.stat_var == _STAT_VAR_B
    assert excinfo.value.code == "internal_error"
    assert _ELECTED in str(excinfo.value)
    assert _UNRECORDED in str(excinfo.value)

    with pytest.raises(DatasetDisagreementError):
        _select_member_facets(requests[1], candidates[1], graph=FakeGraphReader())


def test_an_unrecorded_provenance_beside_the_elected_one_is_left_out():
    request = _member(_STAT_VAR_A)

    kept = dataset_candidates(request, [_facet(_UNRECORDED, obs_count=9), _facet(_ELECTED)])

    assert [candidate.provenance_id for candidate in kept] == [_ELECTED]


def test_a_request_with_no_elected_dataset_keeps_every_candidate():
    request = _member(_STAT_VAR_A, pinned=False)
    candidates = [_facet(_UNRECORDED), _facet(_ELECTED)]

    assert dataset_candidates(request, candidates) == candidates


_CRS = "provenance/OECDCreditorReportingSystem"
_DAC2A = "provenance/OECDDAC2A"


def test_excluding_crs_on_a_role_bearing_member_serves_no_crs_facet():
    """A source-scoped role-bearing probe admits its coverage rows before it ranks facets."""
    coverage = build_coverage_index(
        [
            (_BARE_A, "f-crs", "country/DONOR", _ENTITY, [2014, 2015]),
            (_BARE_A, "f-dac", "country/DONOR", _ENTITY, [2015]),
        ],
        role_names={_BARE_A: RoleNames(donor="donor", recipient="recipient")},
        facet_metadata={
            "f-crs": FacetMetadata(provenance_id=_CRS, measurement_method="M", unit="U"),
            "f-dac": FacetMetadata(provenance_id=_DAC2A, measurement_method="M", unit="U"),
        },
    )
    request = dataclasses.replace(
        _request(stat_var=_STAT_VAR_A, facet_bindings={}, index_sourced=True),
        dataset=ElectedDataset(provenance=_DAC2A, member_provenances=(_CRS, _DAC2A)),
    )

    with bound_catalogue(FakeCatalogue().scoped(frozenset({_DAC2A}))):
        (probe,) = probe_collection_candidates(
            [request],
            availability=NO_SCOPE_READ,
            settle=True,
            graph=FakeGraphReader(),
            coverage=coverage,
        )
        selections = _select_member_facets(request, probe.candidates, graph=FakeGraphReader())

    assert probe.admission == CoverageAdmission(
        pairs_by_facet={"f-dac": frozenset({("country/DONOR", _ENTITY)})},
        role_names=RoleNames(donor="donor", recipient="recipient"),
    )
    assert {candidate.provenance_id for candidate in probe.candidates} == {_DAC2A}
    assert selections.selections["provenance"].ref == _DAC2A
    assert selections.selections["provenance"].dispatch_facet_id == "f-dac"


def test_role_facet_candidates_ignore_source_scope_but_exclude_globally_unavailable_sources():
    """A replacement source pin sees global choices, never a disabled UN or unknown source."""
    served, other, disabled_un, unknown = (
        "provenance/Served",
        "provenance/Other",
        "undata/p/Disabled",
        "provenance/Unknown",
    )
    coverage = build_coverage_index(
        [
            (_BARE_A, "f-served", "country/DONOR", _ENTITY, [2015]),
            (_BARE_A, "f-other", "country/DONOR", _ENTITY, [2015]),
            (_BARE_A, "f-un", "country/DONOR", _ENTITY, [2015]),
            (_BARE_A, "f-unknown", "country/DONOR", _ENTITY, [2015]),
        ],
        role_names={_BARE_A: RoleNames(donor="donor", recipient="recipient")},
        facet_metadata={
            "f-served": FacetMetadata(provenance_id=served),
            "f-other": FacetMetadata(provenance_id=other),
            "f-un": FacetMetadata(provenance_id=disabled_un),
            "f-unknown": FacetMetadata(provenance_id=unknown),
        },
    )
    catalogue = restrictive_catalogue(
        provenances=[
            provenance_row(served, BASE),
            provenance_row(other, BASE),
            provenance_row(disabled_un, UN_SYSTEM, segment="fixture"),
        ],
        un_system_enabled=False,
    ).scoped(frozenset({served}))

    with bound_catalogue(catalogue):
        served_facets = role_bearing_facets(stat_var=_BARE_A, role_bindings={}, coverage=coverage)
        candidates = role_bearing_facet_candidates(
            stat_var=_BARE_A, role_bindings={}, coverage=coverage
        )
        by_donor = role_bearing_facet_candidates_by_donor(
            stat_vars=[_BARE_A], donors=["country/DONOR", "country/NONE"], coverage=coverage
        )

    assert [facet.provenance_id for facet in served_facets or []] == [served]
    assert {facet.provenance_id for facet in candidates or []} == {served, other}
    # Read by donor in one grouped pass, the choices are the same, ranked the same.
    assert by_donor == {_BARE_A: {"country/DONOR": candidates, "country/NONE": []}}


def test_role_facet_candidates_keep_known_provenance_less_method_and_unit():
    """A known unsourced facet remains a valid method or unit candidate."""
    coverage = build_coverage_index(
        [(_BARE_A, "f-unsourced", "country/DONOR", _ENTITY, [2015])],
        role_names={_BARE_A: RoleNames(donor="donor", recipient="recipient")},
        facet_metadata={
            "f-unsourced": FacetMetadata(
                provenance_id=None,
                measurement_method="Reported",
                unit="CurrencyUSD",
            )
        },
    )
    catalogue = restrictive_catalogue(provenances=[], un_system_enabled=False)

    with bound_catalogue(catalogue):
        candidates = role_bearing_facet_candidates(
            stat_var=_BARE_A, role_bindings={}, coverage=coverage
        )

    assert candidates is not None
    assert [
        (facet.provenance_id, facet.measurement_method, facet.unit) for facet in candidates
    ] == [(None, "Reported", "CurrencyUSD")]


def test_scoped_role_coverage_admits_every_origin_per_pair_everywhere_it_is_read():
    """A DFO/base overlap with one base-only pair stays pair-specific for probes and plans, and
    the base facet is admitted at the pair DFO also holds: the election chooses the origin
    (#1271)."""
    dfo, base = "provenance/DFO", "provenance/Base"
    coverage = build_coverage_index(
        [
            (_BARE_A, "f-dfo", "country/NOR", _ENTITY, [2015]),
            (_BARE_A, "f-base", "country/NOR", _ENTITY, [2015]),
            (_BARE_A, "f-base", "country/USA", _ENTITY, [2016]),
        ],
        role_names={_BARE_A: RoleNames(donor="donor", recipient="recipient")},
        facet_metadata={
            "f-dfo": FacetMetadata(provenance_id=dfo, measurement_method="M", unit="U"),
            "f-base": FacetMetadata(provenance_id=base, measurement_method="M", unit="U"),
        },
    )
    catalogue = restrictive_catalogue(
        provenances=[provenance_row(dfo, DFO), provenance_row(base, BASE)]
    ).scoped(frozenset({dfo, base}))

    with bound_catalogue(catalogue):
        probe = coverage_probe(stat_var=_BARE_A, bound={}, coverage=coverage)
        options_facets = role_bearing_facets(stat_var=_BARE_A, role_bindings={}, coverage=coverage)
        served = coverage.series_years(variables=[_BARE_A])

    assert probe.admission == CoverageAdmission(
        pairs_by_facet={
            "f-dfo": frozenset({("country/NOR", _ENTITY)}),
            "f-base": frozenset({("country/NOR", _ENTITY), ("country/USA", _ENTITY)}),
        },
        role_names=RoleNames(donor="donor", recipient="recipient"),
    )
    assert {facet.provenance_id for facet in probe.candidates} == {dfo, base}
    assert {facet.provenance_id for facet in options_facets or []} == {dfo, base}
    assert served is not None
    assert set(served.keys()) == {
        (_BARE_A, _ENTITY, "country/NOR"),
        (_BARE_A, _ENTITY, "country/USA"),
    }
