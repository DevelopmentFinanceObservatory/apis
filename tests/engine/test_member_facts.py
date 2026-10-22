"""Protocol tests for member facts. Graph arcs and fixture index rows must
yield the same five-tuple, constraints, and declared roles.
"""

from __future__ import annotations

from code_index.shared import get_index
from code_index.types import StatVarRow
from search.engine.retrieval.member_facts import ArcsMemberFacts, IndexMemberFacts


def _arcs_from_row(row: StatVarRow) -> dict:
    """Build graph-shaped arcs from the typed fields in `row`."""
    arcs: dict = {
        "measuredProperty": {"nodes": [{"dcid": row.measured_property}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in row.constraint_properties]},
        "observationProperties": {"nodes": [{"dcid": r} for r in row.observation_properties]},
    }
    if row.population_type:
        arcs["populationType"] = {"nodes": [{"dcid": row.population_type}]}
    if row.stat_type:
        arcs["statType"] = {"nodes": [{"dcid": row.stat_type}]}
    if row.measurement_qualifier:
        arcs["measurementQualifier"] = {"nodes": [{"dcid": row.measurement_qualifier}]}
    if row.measurement_denominator:
        arcs["measurementDenominator"] = {"nodes": [{"dcid": row.measurement_denominator}]}
    for prop, value in row.constraint_properties.items():
        arcs[prop] = {"nodes": [{"dcid": value}]}
    return arcs


def _real_crs_row() -> StatVarRow:
    """Return a fixture row with a qualifier and at least 3 constraints."""
    index = get_index()
    for row in index.statvars:
        if row.measurement_qualifier and len(row.constraint_properties) >= 3:
            return row
    raise AssertionError("fixture index carries no StatVarRow rich enough for this test")


def test_read_five_tuple_agrees_between_arcs_and_index_sources():
    row = _real_crs_row()
    arcs_facts = ArcsMemberFacts(arcs=_arcs_from_row(row))
    index_facts = IndexMemberFacts(row=row)

    assert arcs_facts.read_five_tuple() == index_facts.read_five_tuple()


def test_read_constraints_agrees_between_arcs_and_index_sources():
    row = _real_crs_row()
    arcs_facts = ArcsMemberFacts(arcs=_arcs_from_row(row))
    index_facts = IndexMemberFacts(row=row)

    assert arcs_facts.read_constraints() == index_facts.read_constraints()
    assert arcs_facts.read_constraints() == dict(row.constraint_properties)


def test_read_declared_roles_agrees_between_arcs_and_index_sources():
    row = _real_crs_row()
    arcs_facts = ArcsMemberFacts(arcs=_arcs_from_row(row))
    index_facts = IndexMemberFacts(row=row)

    assert arcs_facts.read_declared_roles() == index_facts.read_declared_roles()
    assert arcs_facts.read_declared_roles() == row.observation_properties


def test_index_sourced_member_reports_no_unresolved_constraints():
    """Packaged rows contain resolved facts, so unresolved constraints are empty."""
    row = _real_crs_row()

    assert IndexMemberFacts(row=row).read_unresolved_constraints() == frozenset()


def test_arcs_sourced_member_still_reports_unresolved_constraints():
    """A declared constraint node without a dcid is unresolved."""
    arcs = {
        "constraintProperties": {"nodes": [{"dcid": "crsPurpose"}]},
        "crsPurpose": {"nodes": [{}]},  # present, but no dcid/value: unresolved
    }

    assert ArcsMemberFacts(arcs=arcs).read_unresolved_constraints() == frozenset({"crsPurpose"})
