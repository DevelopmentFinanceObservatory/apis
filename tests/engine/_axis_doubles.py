"""A `GraphReader` double that confirms axis filters against family forward arcs.
It returns materialise's tuple arc shape, reads the arcs of other nodes from a `FakeGraph`,
and delegates other reads to an inner reader.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from graph_generation import GenerationStamp
from materialise import GraphReader
from materialise.graph import UNRESOLVED_NODE, SubjectPage
from tests.fixtures import FakeGraph

_GENERATION_TOKENS = itertools.count()


@dataclass
class AxisConfirmingGraphReader:
    """Confirms family arcs and delegates other reads to `inner` when supplied.

    The reverse query reads the family alone. `graph`, when supplied, answers the arcs of a
    node outside the family, such as a posted member the mint's walk takes as a seed."""

    family_arcs: Mapping[str, Mapping[str, object]]
    inner: GraphReader | None = None
    graph: FakeGraph | None = None
    _token: str = field(
        default_factory=lambda: f"axis-confirming-{next(_GENERATION_TOKENS)}", init=False
    )

    def generation_stamp(self) -> GenerationStamp:
        """A token of this double's own, so two doubles' families never share a member group
        the shared enumerator cached."""
        return GenerationStamp(token=self._token)

    def _asserted_dcids(self, arcs: Mapping[str, object], prop: str) -> set[str]:
        nodes = arcs.get(prop, {}).get("nodes", [])  # type: ignore[union-attr]
        return {n["dcid"] for n in nodes if "dcid" in n}

    def _asserted_values(self, arcs: Mapping[str, object], prop: str) -> tuple[str, ...]:
        nodes = arcs.get(prop, {}).get("nodes", [])  # type: ignore[union-attr]
        return tuple(n["dcid"] if "dcid" in n else n.get("value", UNRESOLVED_NODE) for n in nodes)

    def node_subjects(
        self,
        dcid: str,
        *,
        property: str,
        filters: Mapping[str, str] | None = None,
        max_results: int,
        deadline: float | None = None,
    ) -> SubjectPage:
        matches = []
        for stat_var, arcs in self.family_arcs.items():
            if dcid not in self._asserted_dcids(arcs, property):
                continue
            if all(
                want in self._asserted_dcids(arcs, key) for key, want in (filters or {}).items()
            ):
                matches.append(stat_var)
        ordered = tuple(sorted(matches))
        if len(ordered) > max_results:
            return SubjectPage(dcids=ordered[:max_results], truncated=True)
        return SubjectPage(dcids=ordered, truncated=False)

    def node_arcs_many(
        self,
        dcids: Sequence[str],
        *,
        properties: Sequence[str],
        individual_probe_budget: int | None = None,
        deadline: float | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]] | None]:
        known = [dcid for dcid in dcids if dcid in self.family_arcs]
        unknown = [dcid for dcid in dcids if dcid not in self.family_arcs]
        result: dict[str, dict[str, tuple[str, ...]] | None] = {
            dcid: {prop: self._asserted_values(self.family_arcs[dcid], prop) for prop in properties}
            for dcid in known
        }
        if unknown and self.graph is not None:
            for dcid in unknown:
                arcs = self.graph.node_arcs(dcid)
                result[dcid] = (
                    None
                    if arcs is None
                    else {prop: self._asserted_values(arcs, prop) for prop in properties}
                )
        elif unknown:
            result.update(self._delegate("node_arcs_many", unknown, properties=properties))
        return result

    def node_names_many(
        self,
        dcids: Sequence[str],
        *,
        deadline: float | None = None,
    ) -> dict[str, tuple[str, ...]]:
        """Read names from family arcs; delegate unknown dcids to `inner`."""
        known = [dcid for dcid in dcids if dcid in self.family_arcs]
        unknown = [dcid for dcid in dcids if dcid not in self.family_arcs]
        result: dict[str, tuple[str, ...]] = {
            dcid: self._asserted_values(self.family_arcs[dcid], "name") for dcid in known
        }
        if unknown:
            delegated = self._delegate("node_names_many", unknown)
            assert isinstance(delegated, dict)
            result.update(delegated)
        return result

    def _delegate(self, method: str, *args: object, **kwargs: object) -> object:
        if self.inner is None:
            raise NotImplementedError(
                f"AxisConfirmingGraphReader.{method} has no inner double to delegate to"
            )
        return getattr(self.inner, method)(*args, **kwargs)

    def node_label(self, dcid: str) -> str | None:
        return self._delegate("node_label", dcid)

    def variable_facets_batch(self, *, stat_var: str, entities, deadline: float | None = None):
        return self._delegate(
            "variable_facets_batch", stat_var=stat_var, entities=entities, deadline=deadline
        )

    def declares_facet_defaults(self) -> bool:
        return bool(self._delegate("declares_facet_defaults"))

    def default_facet_policy(self, *, stat_var: str, axis: str):
        return self._delegate("default_facet_policy", stat_var=stat_var, axis=axis)

    def node_arc_values(self, dcid: str, *, properties):
        return self._delegate("node_arc_values", dcid, properties=properties)


def axis_confirming_reader(
    *stat_var_dcids: str, graph: FakeGraph | None = None, inner: GraphReader | None = None
) -> AxisConfirmingGraphReader:
    """Build a reader that checks declared axes against each StatVar's forward arcs.

    `graph` defaults to `FakeGraph`; `inner` supplies other graph methods.
    """
    reader_graph = graph if graph is not None else FakeGraph()
    return AxisConfirmingGraphReader(
        family_arcs={sv: reader_graph.node_arcs(sv) for sv in stat_var_dcids},
        inner=inner,
        graph=reader_graph,
    )


# The India population candidates cover 14 axes across Person/count/measuredValue siblings.
# Axes: age, nativity, employment, religion, race, placeOfResidenceClassification,
# literacyStatus, educationalAttainment, incomeStatus, residenceType, gender,
# armedForcesStatus, placeOfBirth, and maritalStatus.
TOTAL_POPULATION_INDIA_FAMILY = (
    "dc/v78t45118s0bb",
    "Count_Person_Female_ForeignBorn_PlaceOfBirthEurope",
    "Count_Person_Employed_NACE/F",
    "indianCensus/Count_Person_Religion_Muslim",
    "Count_Person_Female_TwoOrMoreRaces",
    "Count_Person_Urban_Female",
    "Count_Person_Literate",
    "Count_Person_EducationalAttainment1StGrade",
    "Count_Person_ResidesInGroupQuarters",
    "Count_Person_WithHealthInsurance",
    "Count_Person_Female_MarriedAndNotSeparated",
)
