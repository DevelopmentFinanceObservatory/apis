"""Keep an index member's facets to the dataset elected for its family.

A mint serves a member of an index-resolved family from its elected dataset alone
(`materialise.facets.dataset_candidates`). Every Search reader of a member's facets (a read
plan, a pin refusal, a facet option, an option's coverage) keeps them by the same test,
`materialise.facets.ElectedDataset.publishes`, through the functions here, so no door
states or admits a facet a mint of the Spec would not serve. A read plan also reads each
kept facet as the mint reads it (`read_in_datasets`): a dataset of several member
provenances stamps its facets with its id, so the plan's elections take them as one table.

A mint refuses a member it serves when only a dataset the code index does not record for
the member publishes its observations at the bound entities
(`ElectedDataset.published`). Each resolver runs that same check once
(`check_elected_datasets`), over the members a mint of its answer serves and every facet
each holds at those entities in any year, as the mint's probe reads them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from materialise.facets import DatasetCandidate, ElectedDataset
from search.engine.clients.graph import Facet
from search.engine.clients.sdmx import RoleFacetCandidate

if TYPE_CHECKING:
    from search.engine.retrieval.member_facts import MemberFacts

type ElectedDatasets = Mapping[str, ElectedDataset]
"""Each member's elected dataset, by StatVar. A member with none is absent."""

NO_DATASETS: ElectedDatasets = {}


class _Sourced(Protocol):
    @property
    def provenance_id(self) -> str | None: ...


class _Read(_Sourced, Protocol):
    @property
    def dataset_id(self) -> str | None: ...


def provenance_axis(candidate: _Read) -> str | None:
    """`candidate`'s value on the provenance axis, as
    `materialise.facets.facet_axis_value` reads a facet's: the elected dataset's id for a
    candidate of a dataset of several member provenances (`read_in_datasets`), else its
    provenance. A conversion converts its rows under this ref, and an answer cites it."""
    return candidate.dataset_id or candidate.provenance_id


def elected_datasets(member_facts: Mapping[str, MemberFacts] | None) -> dict[str, ElectedDataset]:
    """The elected dataset of each member that has one (`MemberFacts.read_dataset`)."""
    return {
        sv: dataset
        for sv, facts in (member_facts or {}).items()
        if (dataset := facts.read_dataset()) is not None
    }


def publishes(datasets: ElectedDatasets, stat_var: str, candidate: _Sourced) -> bool:
    """Whether `stat_var`'s elected dataset publishes `candidate`, or it has none."""
    dataset = datasets.get(stat_var)
    return dataset is None or dataset.publishes(candidate.provenance_id)


def read_in_datasets[C: (Facet, RoleFacetCandidate)](
    datasets: ElectedDatasets, stat_var: str, candidate: C
) -> C:
    """`candidate` as a mint of `stat_var` reads it (`ElectedDataset.read_as_dataset`):
    carrying its elected dataset's id as its `dataset_id` when that dataset has several
    member provenances and publishes it, so `materialise.facets.facet_axis_value` reads one
    provenance for every member's facets. Any other candidate is returned as it is."""
    dataset = datasets.get(stat_var)
    if (
        dataset is None
        or not dataset.several_members
        or not dataset.publishes(candidate.provenance_id)
    ):
        return candidate
    return replace(candidate, dataset_id=dataset.provenance)


def published[C: (Facet, RoleFacetCandidate)](
    datasets: ElectedDatasets, stat_var: str, candidates: Iterable[C]
) -> list[C]:
    """The `candidates` of `stat_var` its elected dataset publishes, in order, each read as
    the dataset's (`read_in_datasets`), so a pin of the dataset's id admits them."""
    return [
        read_in_datasets(datasets, stat_var, c)
        for c in candidates
        if publishes(datasets, stat_var, c)
    ]


def published_by_stat_var[C: (Facet, RoleFacetCandidate)](
    datasets: ElectedDatasets, candidates: Mapping[str, Iterable[C]]
) -> dict[str, list[C]]:
    """`published` over each StatVar's candidates."""
    return {sv: published(datasets, sv, found) for sv, found in candidates.items()}


def check_elected_datasets(
    datasets: ElectedDatasets, candidates: Mapping[str, Iterable[DatasetCandidate]]
) -> None:
    """Raise the mint's `DatasetDisagreementError` for a member of `candidates` that its
    mint refuses (`ElectedDataset.published`). `candidates` maps each member a mint of the
    answer serves to every facet it holds at the bound entities, over every year."""
    for stat_var, found in candidates.items():
        if (dataset := datasets.get(stat_var)) is not None:
            dataset.published(stat_var, found)
