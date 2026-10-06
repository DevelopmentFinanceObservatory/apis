"""Tests joint election, forced values, dropped axes, siblings, and role filtering.

Synthetic axes cover constraint/basis behavior absent from the deployed CRS declarations.
"""

from __future__ import annotations

from code_index import CodeIndex, IndexStamp, StatVarRow
from code_index.types import MeasureAxis
from search.engine.retrieval.bound_values import elect_measure_family
from search.engine.shape import ShapeDraft
from tests.engine._index_slots import crs_flow_slot_keys, query_for

_FIVE_TUPLE = {
    "pop_type_dcid": "Widget",
    "meas_prop_dcid": "count",
    "stat_type_dcid": "measuredValue",
    "meas_qual_dcid": None,
    "meas_denom_dcid": None,
}


def _row(
    dcid: str,
    *,
    population_type: str | None = None,
    observation_properties: tuple[str, ...] = (),
    **constraints: str,
) -> StatVarRow:
    return StatVarRow(
        dcid=dcid,
        population_type=population_type or _FIVE_TUPLE["pop_type_dcid"],
        measured_property=_FIVE_TUPLE["meas_prop_dcid"],
        stat_type=_FIVE_TUPLE["stat_type_dcid"],
        measurement_qualifier=_FIVE_TUPLE["meas_qual_dcid"],
        measurement_denominator=_FIVE_TUPLE["meas_denom_dcid"],
        observation_properties=observation_properties,
        constraint_properties=constraints,
        provenances=("provenance/Test",),
    )


def _index(statvars: list[StatVarRow]) -> CodeIndex:
    return CodeIndex(
        stamp=IndexStamp(
            graph_endpoint="test",
            exported_at="2026-01-01T00:00:00Z",
            code_counts={},
            statvar_shape_counts={},
            content_hash="test-hash",
        ),
        declarations=(),
        codes=(),
        statvars=statvars,
    )


def _shape(*, declared_roles: tuple[str, ...] | None = None) -> ShapeDraft:
    return ShapeDraft(
        shape_id="widget",
        label="widget",
        pop_type_dcid=_FIVE_TUPLE["pop_type_dcid"],
        meas_prop_dcid=_FIVE_TUPLE["meas_prop_dcid"],
        stat_type_dcid=_FIVE_TUPLE["stat_type_dcid"],
        meas_qual_dcid=_FIVE_TUPLE["meas_qual_dcid"],
        meas_denom_dcid=_FIVE_TUPLE["meas_denom_dcid"],
        slot_keys=(),
        declared_roles=declared_roles,
    )


_MATERIAL = MeasureAxis(
    axis="material", kind="constraint", role="free", prefer=("Plastic", "Metal")
)
_FINISH = MeasureAxis(axis="finish", kind="constraint", role="free", prefer=("Matte", "Glossy"))
_WARRANTY = MeasureAxis(axis="warranty", kind="constraint", role="free", prefer=("Yes", "No"))


def test_most_significant_axis_decides_over_a_later_one():
    # SV1 wins on material (the more significant axis, listed first) but
    # loses on finish; SV2 is the reverse. material decides regardless.
    rows = [
        _row("SV1", material="Plastic", finish="Glossy"),
        _row("SV2", material="Metal", finish="Matte"),
    ]
    index = _index(rows)

    election = elect_measure_family(_shape(), index, (_MATERIAL, _FINISH), query_for(_shape(), {}))

    assert election is not None
    assert election.forced == (("material", "Plastic"), ("finish", "Glossy"))


def test_axis_already_bound_is_not_elected():
    rows = [
        _row("SV1", material="Plastic", finish="Glossy"),
        _row("SV2", material="Metal", finish="Matte"),
    ]
    index = _index(rows)

    election = elect_measure_family(
        _shape(), index, (_MATERIAL, _FINISH), query_for(_shape(), {"material": ["Metal"]})
    )

    assert election is not None
    # Exclude the query-bound axis even though its matching row has a value.
    assert election.forced == (("finish", "Matte"),)
    # Query-bound axes appear in neither result.
    assert election.dropped == ()


def test_axis_whose_elected_partition_carries_none_yields_no_pair():
    # Missing warranty values leave the axis open; no value can be forced.
    rows = [
        _row("SV1", material="Plastic"),
        _row("SV2", material="Metal"),
    ]
    index = _index(rows)

    election = elect_measure_family(
        _shape(), index, (_MATERIAL, _WARRANTY), query_for(_shape(), {})
    )

    assert election is not None
    assert election.forced == (("material", "Plastic"),)
    # warranty is reported dropped, not merely absent from forced -- and
    # material (which was forced) does not also show up here.
    assert election.dropped == ("warranty",)


def test_empty_populated_set_fails_open_rather_than_electing_unscoped():
    # No matching row means no election or forced value.
    rows = [
        _row("SV1", material="Plastic", finish="Glossy"),
        _row("SV2", material="Metal", finish="Matte"),
    ]
    index = _index(rows)

    election = elect_measure_family(
        _shape(), index, (_MATERIAL, _FINISH), query_for(_shape(), {"material": ["Chrome"]})
    )

    assert election is None

    # Without the unmatched filter, the same index elects `finish`.
    unscoped = elect_measure_family(_shape(), index, (_MATERIAL, _FINISH), query_for(_shape(), {}))
    assert unscoped is not None
    assert unscoped.forced != ()


def test_joint_election_never_returns_a_combination_no_row_carries():
    # Axis values correlate, so independent preferences would invent Air+Pending.
    # Joint election returns an observed pair; preference order selects Air+Delivered.
    mode = MeasureAxis(axis="packageMode", kind="constraint", role="free", prefer=("Air", "Sea"))
    status = MeasureAxis(
        axis="packageStatus", kind="constraint", role="free", prefer=("Pending", "Delivered")
    )
    rows = [
        _row("SV_AIR", packageMode="Air", packageStatus="Delivered"),
        _row("SV_SEA", packageMode="Sea", packageStatus="Pending"),
    ]
    index = _index(rows)

    election = elect_measure_family(_shape(), index, (mode, status), query_for(_shape(), {}))

    assert election is not None
    assert election.forced == (("packageMode", "Air"), ("packageStatus", "Delivered"))


def test_forced_and_dropped_partition_the_open_constraint_axes_disjointly():
    # The query binds material, finish is forced, and unpopulated warranty is dropped.
    rows = [
        _row("SV1", material="Plastic", finish="Glossy"),
        _row("SV2", material="Metal", finish="Matte"),
    ]
    index = _index(rows)

    election = elect_measure_family(
        _shape(),
        index,
        (_MATERIAL, _FINISH, _WARRANTY),
        query_for(_shape(), {"material": ["Plastic"]}),
    )

    assert election is not None
    assert election.forced == (("finish", "Glossy"),)
    assert election.dropped == ("warranty",)
    assert not (set(dict(election.forced)) & set(election.dropped))


# Tests sibling selection on shape axes and synthetic constraint/basis axes.
# CRS does not declare a constraint/basis axis.

_POPULATION = MeasureAxis(
    axis="populationType", kind="shape", role="free", prefer=("Widget", "Gadget")
)
_GRADE = MeasureAxis(axis="grade", kind="constraint", role="basis", prefer=("A", "B"))
_FAMILY_FINISH = MeasureAxis(
    axis="finish", kind="constraint", role="free", prefer=("Matte", "Glossy")
)
_FAMILY_AXES = (_POPULATION, _GRADE, _FAMILY_FINISH)


def _family_rows() -> list[StatVarRow]:
    return [
        _row("SV_W_A_MATTE", population_type="Widget", grade="A", finish="Matte"),
        _row("SV_W_A_GLOSSY", population_type="Widget", grade="A", finish="Glossy"),
        _row("SV_W_B_MATTE", population_type="Widget", grade="B", finish="Matte"),
        _row("SV_G_A_MATTE", population_type="Gadget", grade="A", finish="Matte"),
        _row("SV_G_B_MATTE", population_type="Gadget", grade="B", finish="Matte"),
        _row("SV_W_A_NONE", population_type="Widget", grade="A"),
    ]


def test_siblings_differ_only_on_free_axes_and_exclude_the_elected_partition():
    # Widget, grade A, and preferred finish Matte form the elected partition.
    index = _index(_family_rows())

    election = elect_measure_family(_shape(), index, _FAMILY_AXES, query_for(_shape(), {}))

    assert election is not None
    assert election.elected.values == (
        ("populationType", "Widget"),
        ("grade", "A"),
        ("finish", "Matte"),
    )
    sibling_values = {s.values for s in election.siblings}
    # Both same-grade partitions differing only on free finish are siblings.
    assert (("populationType", "Widget"), ("grade", "A"), ("finish", "Glossy")) in sibling_values
    assert (("populationType", "Widget"), ("grade", "A"), ("finish", None)) in sibling_values
    # A free shape axis may differ too, provided basis grade still matches.
    assert (("populationType", "Gadget"), ("grade", "A"), ("finish", "Matte")) in sibling_values
    # A basis-grade mismatch excludes a partition, regardless of free-axis matches.
    assert (("populationType", "Widget"), ("grade", "B"), ("finish", "Matte")) not in sibling_values
    assert (("populationType", "Gadget"), ("grade", "B"), ("finish", "Matte")) not in sibling_values
    assert election.elected.values not in sibling_values
    assert len(election.siblings) == 3


def test_full_set_still_offers_the_other_measure_value_as_a_sibling():
    # The Glossy winner retains its Matte sibling from the full candidate set.
    index = _index(_family_rows())

    election = elect_measure_family(
        _shape(), index, _FAMILY_AXES, query_for(_shape(), {"finish": ["Glossy"]})
    )

    assert election is not None
    assert election.elected.values == (
        ("populationType", "Widget"),
        ("grade", "A"),
        ("finish", "Glossy"),
    )
    sibling_values = {s.values for s in election.siblings}
    assert (("populationType", "Widget"), ("grade", "A"), ("finish", "Matte")) in sibling_values


# ---------------------------------------------------------------------------
# Non-measure bindings and no-consistent-partition.
# ---------------------------------------------------------------------------


def test_a_bound_non_measure_property_removes_a_partition_outside_its_scope():
    # Binding the non-measure `region=West` removes the East row and its sole partition.
    rows = [
        _row("SV1", finish="Matte", region="West"),
        _row("SV2", finish="Glossy", region="East"),
    ]
    index = _index(rows)

    unscoped = elect_measure_family(_shape(), index, (_FINISH,), query_for(_shape(), {}))
    assert unscoped is not None
    assert len(unscoped.siblings) == 1

    scoped = elect_measure_family(
        _shape(), index, (_FINISH,), query_for(_shape(), {"region": ["West"]})
    )
    assert scoped is not None
    assert scoped.elected.values == (("finish", "Matte"),)
    assert scoped.siblings == ()


def test_no_consistent_partition_yields_no_forced_pair_and_no_sibling():
    # The shape is Widget, while every indexed row is Gadget, leaving no consistent partition.
    rows = [
        _row("SV_G_A", population_type="Gadget", grade="A"),
        _row("SV_G_B", population_type="Gadget", grade="B"),
    ]
    index = _index(rows)

    election = elect_measure_family(_shape(), index, (_POPULATION, _GRADE), query_for(_shape(), {}))

    assert election is None


# ---------------------------------------------------------------------------
# The real packaged index: 4 populated partitions, 2 siblings for the
# default (gross nominal disbursements) election, grant equivalents never
# among them.
# ---------------------------------------------------------------------------


def test_real_index_elects_gross_nominal_disbursements_with_two_siblings():
    from code_index.shared import get_index, get_measure_index

    index = get_index()
    axes = get_measure_index().axes
    flow_row = next(row for row in index.statvars if row.population_type == "FinancialFlow")
    shape = ShapeDraft(
        shape_id="crs-amount",
        label="crs-amount",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid=flow_row.stat_type,
        meas_qual_dcid="Nominal",
        meas_denom_dcid=None,
        slot_keys=crs_flow_slot_keys(),
    )

    election = elect_measure_family(shape, index, axes, query_for(shape, {}))

    assert election is not None
    elected_values = dict(election.elected.values)
    assert elected_values["populationType"] == "FinancialFlow"
    assert elected_values["measuredProperty"] == "amount"
    assert elected_values["measurementQualifier"] == "Nominal"
    assert elected_values["nettingBasis"] == "Gross"
    assert election.forced == (("financialFlowType", "Disbursement"), ("nettingBasis", "Gross"))
    # financialObligationType belongs to FinancialObligation rows, never a
    # FinancialFlow one -- no member of this family declares it at all.
    assert election.dropped == ("financialObligationType",)
    assert len(election.siblings) == 2
    assert all(dict(s.values)["measuredProperty"] != "grantEquivalent" for s in election.siblings)


# ---------------------------------------------------------------------------
# Role matching: a candidate row must declare the shape's own roles.
# ---------------------------------------------------------------------------


def test_row_declaring_different_roles_is_neither_elected_nor_a_sibling():
    rows = [
        _row("SV_MATCH", observation_properties=("financingRecipient",), finish="Matte"),
        _row("SV_OTHER_ROLE", observation_properties=("financingProvider",), finish="Glossy"),
    ]
    index = _index(rows)
    shape = _shape(declared_roles=("financingRecipient",))

    election = elect_measure_family(shape, index, (_FINISH,), query_for(shape, {}))

    assert election is not None
    assert election.elected.values == (("finish", "Matte"),)
    assert election.siblings == ()


def test_row_declaring_the_shapes_own_roles_is_elected():
    # Same two rows as above, this time electing the role the shape itself
    # carries -- the Glossy row is now the only candidate, proving the
    # filter selects on role identity rather than merely excluding one side.
    rows = [
        _row("SV_MATCH", observation_properties=("financingRecipient",), finish="Matte"),
        _row("SV_OTHER_ROLE", observation_properties=("financingProvider",), finish="Glossy"),
    ]
    index = _index(rows)
    shape = _shape(declared_roles=("financingProvider",))

    election = elect_measure_family(shape, index, (_FINISH,), query_for(shape, {}))

    assert election is not None
    assert election.elected.values == (("finish", "Glossy"),)
    assert election.siblings == ()


def test_empty_observation_properties_reads_as_observation_about():
    # A row's own observation_properties=() is the index's spelling of a
    # plain subject-only member -- the same ("observationAbout",) default
    # IndexMemberFacts.read_declared_roles applies elsewhere, so a shape
    # declaring that role still matches it.
    rows = [_row("SV_SUBJECT", observation_properties=(), finish="Matte")]
    index = _index(rows)
    shape = _shape(declared_roles=("observationAbout",))

    election = elect_measure_family(shape, index, (_FINISH,), query_for(shape, {}))

    assert election is not None
    assert election.elected.members == ("SV_SUBJECT",)


def test_declared_roles_none_leaves_role_filtering_off():
    # A ShapeDraft built outside derive_shapes (declared_roles=None) carries
    # no role set to compare against, so rows are selected exactly as they
    # were before roles were considered at all -- both rows here stay
    # candidates despite declaring different roles from each other. The
    # query binds no role and neither row is subject-only, so the family
    # election keeps both.
    rows = [
        _row("SV_PROVIDER", observation_properties=("financingProvider",), finish="Matte"),
        _row("SV_ROLE", observation_properties=("financingRecipient",), finish="Glossy"),
    ]
    index = _index(rows)
    shape = _shape(declared_roles=None)

    election = elect_measure_family(shape, index, (_FINISH,), query_for(shape, {}))

    assert election is not None
    assert election.elected.values == (("finish", "Matte"),)
    assert len(election.siblings) == 1
