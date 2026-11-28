"""Keep an index member's facets to the dataset elected for its family.

A mint serves a member of an index-resolved family from its elected dataset alone
(`materialise.facets.dataset_candidates`). Every Search reader of a member's facets (a read
plan, a pin refusal, a facet option, an option's coverage) keeps them by the same test,
`materialise.facets.ElectedDataset.publishes`, through the functions here, so no door
states or admits a facet a mint of the Spec would not serve. Every facet a reader keeps is
read as the mint reads it (`materialise.facets.read_as_datasets`): a facet of a declared
dataset's member provenance carries the dataset's id, so the elections take the members as one
table. Every Search reader of a candidate's provenance axis reads `provenance_axis`.

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

from code_index.shared import get_index
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


def provenance_axis(candidate: _Sourced) -> str | None:
    """`candidate`'s value on the provenance axis, as `materialise.facets.facet_axis_value`
    reads a facet once read (`read_as_datasets`): the id of the declared dataset its
    provenance is a member of in the bound release's code index (`CodeIndex.dataset_of`),
    which is the provenance itself for a plain table. A function of the provenance alone, so a
    facet, a role candidate and a bulk candidate read alike, stamped or not. An answer cites
    this ref, a series states it, and a conversion converts its rows under it."""
    provenance = candidate.provenance_id
    return None if provenance is None else get_index().dataset_of(provenance)


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


def published[C: (Facet, RoleFacetCandidate)](
    datasets: ElectedDatasets, stat_var: str, candidates: Iterable[C]
) -> list[C]:
    """The `candidates` of `stat_var` its elected dataset publishes, in order. A facet is read
    as a mint reads it (`materialise.facets.read_as_datasets`), carrying its dataset's id, so a
    pin of that id admits it; a role candidate's provenance axis is `provenance_axis`."""
    return [_read(c) for c in candidates if publishes(datasets, stat_var, c)]


def _read[C: (Facet, RoleFacetCandidate)](candidate: C) -> C:
    """A facet as a mint reads it (`materialise.facets.read_as_datasets`), and a role
    candidate as it is."""
    if isinstance(candidate, RoleFacetCandidate):
        return candidate
    dataset = provenance_axis(candidate)
    if dataset == candidate.provenance_id or dataset == candidate.dataset_id:
        return candidate
    return replace(candidate, dataset_id=dataset)


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
