"""A dataset of several member provenances elects as one table.

The synthetic `dataset/Ledger` splits one table's series between two provenances, as a
register filed by publisher does. Facet selection and group agreement key on the dataset, so
both members' facets are selected, one group forms, and the dataset's declaration on absent
members reads for the whole selection. A plain table, its own sole member, is the control.
"""

from __future__ import annotations

import dataclasses

import pytest

from materialise import EnrichmentLookup, materialise_group
from materialise.facets import (
    ElectedDataset,
    SeriesCounts,
    joint_dispatch_facet_ids,
    rank_candidates,
)
from materialise.generation import _select_member_facets, reads_absent_as_zero
from materialise.graph import Facet
from materialise.group_facets import select_group_facets
from series_coverage.schema import FacetMetadata, RoleNames
from tests.engine._fake_coverage import build_coverage_index

from ._group_support import (
    _BARE_A,
    _BARE_B,
    _CSV_HEADER,
    _ENTITY,
    _STAT_VAR_A,
    _STAT_VAR_B,
    _STAT_VAR_C,
    _csv_row,
    _request,
    _sdmx_client_for,
)
from .conftest import FakeGraphReader

_LEDGER = "dataset/Ledger"
_NORTH = "provenance/LedgerNorth"
_SOUTH = "provenance/LedgerSouth"
_PLAIN = "provenance/Plain"

_LEDGER_DATASET = ElectedDataset(
    provenance=_LEDGER, provenances=(_NORTH, _SOUTH), recorded_provenances=(_NORTH, _SOUTH)
)
_PLAIN_DATASET = ElectedDataset(
    provenance=_PLAIN, provenances=(_PLAIN,), recorded_provenances=(_PLAIN,)
)


def _facet(
    provenance: str, *, obs_count: int = 1, unit: str = "UNIT", method: str = "METH"
) -> Facet:
    suffix = "" if (unit, method) == ("UNIT", "METH") else f"-{unit}-{method}"
    return Facet(
        facet_id=f"facet-{provenance}{suffix}",
        provenance_id=provenance,
        measurement_method=method,
        unit=unit,
        obs_count=obs_count,
        earliest_date="2015",
        latest_date="2015",
        observation_period="P1Y",
    )


def _member(stat_var: str, dataset: ElectedDataset = _LEDGER_DATASET):
    request = _request(stat_var=stat_var, facet_bindings={}, index_sourced=True)
    return dataclasses.replace(request, dataset=dataset)


def test_the_dataset_of_provenances_is_built_from_the_index_member_map():
    members = {_LEDGER: (_NORTH, _SOUTH)}

    dataset = ElectedDataset.of(
        _LEDGER,
        recorded=(_LEDGER, _PLAIN),
        provenances_of=lambda dataset: members.get(dataset, (dataset,)),
    )

    assert dataset == ElectedDataset(
        provenance=_LEDGER,
        provenances=(_NORTH, _SOUTH),
        recorded_provenances=(_NORTH, _SOUTH, _PLAIN),
    )


@pytest.mark.parametrize(
    ("dataset", "provenances", "ref", "dispatched"),
    [
        pytest.param(
            _LEDGER_DATASET,
            (_NORTH, _SOUTH),
            _LEDGER,
            (f"facet-{_NORTH}", f"facet-{_SOUTH}"),
            id="two_members",
        ),
        pytest.param(_PLAIN_DATASET, (_PLAIN,), _PLAIN, (f"facet-{_PLAIN}",), id="plain_table"),
    ],
)
def test_a_member_selects_its_dataset_and_dispatches_every_member_provenance(
    dataset, provenances, ref, dispatched
):
    """The members hold different series of one variable, so the election reads them as one
    selection with no disagreement to report, and the dispatch over the facets as probed
    serves both."""
    request = _member(_STAT_VAR_A, dataset)
    candidates = [_facet(provenance, obs_count=3 - i) for i, provenance in enumerate(provenances)]

    election = _select_member_facets(
        request,
        candidates,
        graph=FakeGraphReader(),
        breadth={f"facet-{p}": 3 - i for i, p in enumerate(provenances)},
    )

    selection = election.selections["provenance"]
    assert selection.ref == ref
    assert selection.selection_reason in ("sole_candidate", "most_observations")
    assert selection.ranked_values == ()
    assert joint_dispatch_facet_ids(candidates, election, member=0) == dispatched


def test_members_held_under_different_provenances_of_one_dataset_form_one_group():
    """Member A holds its series under one provenance and member B under the other. Keyed on
    provenance they would form two groups and leave one member unserved."""
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B)]
    candidates = [[_facet(_NORTH, obs_count=2)], [_facet(_SOUTH)]]

    group = select_group_facets(requests, candidates, graph=FakeGraphReader())

    assert group.election.selections["provenance"].ref == _LEDGER
    assert group.unserved == ()
    assert group.dispatch_ids == ((f"facet-{_NORTH}",), (f"facet-{_SOUTH}",))


def test_the_unit_both_member_provenances_hold_serves_the_group():
    """Members A and B hold the dataset's series in one unit under different provenances,
    and member C a larger facet in another unit. The key every member but C holds is the
    dataset's in the first unit, so it serves A and B and leaves C out."""
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B), _member(_STAT_VAR_C)]
    candidates = [
        [_facet(_NORTH)],
        [_facet(_SOUTH)],
        [_facet(_NORTH, obs_count=5, unit="OTHER")],
    ]

    group = select_group_facets(requests, candidates, graph=FakeGraphReader())

    assert group.election.selections["unit"].ref == "UNIT"
    assert group.unserved == (2,)


def test_under_national_currency_a_method_counts_the_series_of_both_member_provenances():
    """Each place's own currency leaves the unit out of the selection, so a selection counts
    the series of all its facets (`rank_candidates`). The method both provenances hold serves
    three plus two series, more than the four the other method holds under one."""
    candidates = _LEDGER_DATASET.read_as_dataset(
        [_facet(_NORTH), _facet(_SOUTH), _facet(_NORTH, method="OTHER")]
    )
    counts = {f"facet-{_NORTH}": 3, f"facet-{_SOUTH}": 2, f"facet-{_NORTH}-UNIT-OTHER": 4}

    ranked = rank_candidates(
        candidates,
        window=None,
        breadth=SeriesCounts(counts=counts, union_of=lambda ids: sum(counts[i] for i in ids)),
        national_currency=True,
    )

    assert ranked[0].measurement_method == "METH"


def test_a_group_of_one_dataset_reads_the_dataset_declaration_on_absent_members():
    requests = [_member(_STAT_VAR_A), _member(_STAT_VAR_B)]
    candidates = [[_facet(_NORTH, obs_count=2)], [_facet(_SOUTH)]]
    selections = select_group_facets(
        requests, candidates, graph=FakeGraphReader()
    ).election.selections
    read: list[str] = []

    def absent_reads_zero(dataset: str) -> bool:
        read.append(dataset)
        return dataset == _LEDGER

    assert reads_absent_as_zero(selections, absent_reads_zero)
    assert read == [_LEDGER]


def test_a_group_of_one_dataset_mints_every_member_provenance_rows():
    coverage = build_coverage_index(
        [
            (_BARE_A, f"facet-{_NORTH}", "country/DONOR", _ENTITY, [2015]),
            (_BARE_B, f"facet-{_SOUTH}", "country/DONOR", _ENTITY, [2015]),
        ],
        role_names={
            bare: RoleNames(donor="donor", recipient="recipient") for bare in (_BARE_A, _BARE_B)
        },
        facet_metadata={
            f"facet-{provenance}": FacetMetadata(
                provenance_id=provenance,
                measurement_method="METH",
                unit="UNIT",
                observation_period="P1Y",
            )
            for provenance in (_NORTH, _SOUTH)
        },
    )
    sdmx, _requests = _sdmx_client_for(
        {
            bare: _CSV_HEADER
            + _csv_row(
                stat_var=bare,
                year="2015",
                value="10",
                unit="UNIT",
                method="METH",
                provenance=provenance,
                facet_id=f"facet-{provenance}",
            )
            for bare, provenance in ((_BARE_A, _NORTH), (_BARE_B, _SOUTH))
        }
    )
    graph = FakeGraphReader()

    result = materialise_group(
        [_member(_STAT_VAR_A), _member(_STAT_VAR_B)],
        sdmx=sdmx,
        graph=graph,
        enrichment=EnrichmentLookup(graph),
        parties={},
        materialiser_build="build-123",
        capability_version="cap-v1",
        coverage=coverage,
    )

    counts = {entry["ref"]: entry["observation_count"] for entry in result.trace.resolved_stat_vars}
    assert counts == {_STAT_VAR_A: 1, _STAT_VAR_B: 1}
    assert not [w for w in result.warnings if w["code"] == "coverage_below_scope"]
    (provenance,) = [entry for entry in result.trace.facets if entry["axis"] == "provenance"]
    assert provenance["ref"] == _LEDGER
