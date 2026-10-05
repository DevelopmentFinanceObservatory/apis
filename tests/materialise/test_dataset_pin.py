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
    ElectedDataset,
    dataset_candidates,
)
from materialise.generation import _select_member_facets
from materialise.graph import Facet
from materialise.group_facets import select_group_facets

from ._group_support import _STAT_VAR_A, _STAT_VAR_B, _request
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

    selections = select_group_facets(requests, candidates, graph=FakeGraphReader())

    assert selections["provenance"].ref == _ELECTED


def test_the_same_group_without_an_election_refuses_for_mixed_provenance():
    requests = [_member(_STAT_VAR_A, pinned=False), _member(_STAT_VAR_B, pinned=False)]
    candidates = [[_facet(_ELECTED)], [_facet(_OTHER, obs_count=9)]]

    with pytest.raises(FacetNotAvailableError, match="no probed facet serves every member"):
        select_group_facets(requests, candidates, graph=FakeGraphReader())


def test_a_member_selects_the_elected_provenance_over_a_larger_facet_of_another_dataset():
    request = _member(_STAT_VAR_A, recorded=(_ELECTED, _OTHER))
    candidates = [_facet(_OTHER, obs_count=9), _facet(_ELECTED)]

    selections = _select_member_facets(request, candidates, graph=FakeGraphReader())

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
    )

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
    )

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
