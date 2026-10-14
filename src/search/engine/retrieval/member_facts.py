"""Unify recall-sourced arcs and index-sourced rows behind one member-facts API.

Both resolvers use the same readers. Index rows use typed fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from code_index.types import StatVarRow
from materialise.facets import ElectedDataset
from search.engine.retrieval.facts import (
    FiveTuple,
    read_constraint_value_names,
    read_constraints,
    read_declared_constraint_properties,
    read_declared_roles,
    read_five_tuple,
    read_name,
    read_unresolved_constraints,
)


class MemberFacts(Protocol):
    """Read one member's identity, constraints, and declared roles."""

    def read_five_tuple(self) -> FiveTuple: ...
    def constraints_known(self) -> bool: ...
    def read_constraints(self) -> dict[str, str]: ...
    def read_declared_constraint_properties(self) -> tuple[str, ...]: ...
    def read_unresolved_constraints(self) -> frozenset[str]: ...
    def read_constraint_value_names(self) -> dict[str, str]: ...
    def read_declared_roles(self) -> tuple[str, ...]: ...
    def read_dataset(self) -> ElectedDataset | None: ...


def asserted_constraint_properties(facts: MemberFacts) -> frozenset[str] | None:
    """Return every constraint property the member asserts, with or without a resolved value.

    ``None`` when the member's constraints are unknown, which differs from the empty
    set of a member known to assert none.
    """
    if not facts.constraints_known():
        return None
    return frozenset(facts.read_constraints()) | facts.read_unresolved_constraints()


def member_name(facts: MemberFacts) -> str | None:
    """The graph name of a member whose fetched arcs `facts` holds (`ArcsMemberFacts`).
    None for a member with no name arc and for a packaged index row, which carries none."""
    return read_name(facts.arcs) if isinstance(facts, ArcsMemberFacts) else None


@dataclass(frozen=True, slots=True)
class ArcsMemberFacts:
    """Read a member's fetched graph arcs.

    `dataset` is set for a posted member that is a row of the family the code index elects
    for its identity (`bound_values.shape_member_facts`), so a resubmit reads it from the
    dataset its first answer and a mint read it from."""

    arcs: dict
    dataset: ElectedDataset | None = None

    def read_five_tuple(self) -> FiveTuple:
        return read_five_tuple(self.arcs)

    def constraints_known(self) -> bool:
        """An empty mapping holds no facts. A full ``->*`` read lists only the arcs a
        node has, so a non-empty one without ``constraintProperties`` asserts none."""
        return bool(self.arcs)

    def read_constraints(self) -> dict[str, str]:
        return read_constraints(self.arcs)

    def read_declared_constraint_properties(self) -> tuple[str, ...]:
        return read_declared_constraint_properties(self.arcs)

    def read_unresolved_constraints(self) -> frozenset[str]:
        return read_unresolved_constraints(self.arcs)

    def read_constraint_value_names(self) -> dict[str, str]:
        return read_constraint_value_names(self.arcs)

    def read_declared_roles(self) -> tuple[str, ...]:
        return read_declared_roles(self.arcs)

    def read_dataset(self) -> ElectedDataset | None:
        """None for a recall-sourced member, which belongs to no elected dataset."""
        return self.dataset


@dataclass(frozen=True, slots=True)
class IndexMemberFacts:
    """Read a packaged index row. Rows have no unresolved-arc state.

    `dataset` is the provenance of the dataset elected for the family the row was read
    into (`code_index.members.DatasetFamily.provenance`), or None for a row read outside
    an election."""

    row: StatVarRow
    dataset: str | None = None

    def read_five_tuple(self) -> FiveTuple:
        return FiveTuple(
            pop_type_dcid=self.row.population_type or "",
            meas_prop_dcid=self.row.measured_property,
            stat_type_dcid=self.row.stat_type or "",
            meas_qual_dcid=self.row.measurement_qualifier,
            meas_denom_dcid=self.row.measurement_denominator,
            # Index rows come from resolved graph reads. shape_member_facts
            # excludes unresolved-denominator shapes before index sourcing.
            meas_denom_unresolved=False,
        )

    def constraints_known(self) -> bool:
        return True

    def read_constraints(self) -> dict[str, str]:
        return dict(self.row.constraint_properties)

    def read_declared_constraint_properties(self) -> tuple[str, ...]:
        return tuple(self.row.constraint_properties)

    def read_unresolved_constraints(self) -> frozenset[str]:
        return frozenset()

    def read_constraint_value_names(self) -> dict[str, str]:
        # The code index labels every value a row carries.
        return {}

    def read_declared_roles(self) -> tuple[str, ...]:
        return self.row.observation_properties or ("observationAbout",)

    def read_dataset(self) -> ElectedDataset | None:
        """The elected dataset as a mint of the member carries it
        (`datasets_api.adapter.resolve_definition_members`)."""
        if self.dataset is None:
            return None
        return ElectedDataset(provenance=self.dataset, member_provenances=self.row.provenances)
