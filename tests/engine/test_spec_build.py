"""Tests denominator refs and unresolved flags in the wire `Shape`."""

from __future__ import annotations

import pytest

from code_index.members import index_family
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import identity_query, source_binding
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import (
    NamedGroup,
    _resolved_stat_vars,
    _shape_stating_identity_family,
    build_shape_model,
    build_spec,
    build_stat_vars,
    pair_slot_drafts,
    stat_var_from_resolved,
)
from search.engine.spec_id import mint_identity
from search.models import (
    BindingUnbound,
    BindingValue,
    CoverageBare,
    Entity,
    EntityRoleDirectional,
    GraphRef,
    IdentityEntityGroup,
    ResolvedStatVar,
    Slot,
    SlotKey,
    SlotValue,
    StatVar,
)
from statvar_members import member_digest
from tests.code_index.aid_identity import (
    CRS_MEMBER_COUNT,
    IMPUTED_PROVENANCE,
    education_aid_identity,
)


def _shape_draft(*, meas_denom_dcid: str | None, meas_denom_unresolved: bool) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="EconomicActivity",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=meas_denom_dcid,
        slot_keys=(),
        meas_denom_unresolved=meas_denom_unresolved,
    )


class TestBuildShapeModelDenominator:
    def test_dcid_bearing_unresolved_denominator_yields_its_ref_and_the_flag(self):
        """A dcid-bearing ProvisionalNode target (meas_denom_dcid set and
        meas_denom_unresolved True) surfaces its dcid on the wire ref same as
        any other denominator. measurement_denominator_unresolved is the
        signal that the graph defines no real StatVar for that dcid. Only
        the identity core (engine.shape.identity_shape_for) drops the dcid."""
        shape_draft = _shape_draft(meas_denom_dcid="PerCapita", meas_denom_unresolved=True)

        shape = build_shape_model(
            shape_draft, slot_keys=[], five_tuple_refs={}, member_count=1, members=None
        )

        assert shape.measurement_denominator == GraphRef(dcid="PerCapita", label="PerCapita")
        assert shape.measurement_denominator_unresolved is True

    def test_dcid_less_unresolved_denominator_yields_no_ref_but_the_flag(self):
        shape_draft = _shape_draft(meas_denom_dcid=None, meas_denom_unresolved=True)

        shape = build_shape_model(
            shape_draft, slot_keys=[], five_tuple_refs={}, member_count=1, members=None
        )

        assert shape.measurement_denominator is None
        assert shape.measurement_denominator_unresolved is True

    def test_resolved_denominator_yields_its_graphref(self):
        shape_draft = _shape_draft(meas_denom_dcid="Count_Person", meas_denom_unresolved=False)

        shape = build_shape_model(
            shape_draft, slot_keys=[], five_tuple_refs={}, member_count=1, members=None
        )

        assert shape.measurement_denominator == GraphRef(dcid="Count_Person", label="Count_Person")
        assert shape.measurement_denominator_unresolved is False

    def test_absent_denominator_yields_no_ref_and_no_flag(self):
        shape_draft = _shape_draft(meas_denom_dcid=None, meas_denom_unresolved=False)

        shape = build_shape_model(
            shape_draft, slot_keys=[], five_tuple_refs={}, member_count=1, members=None
        )

        assert shape.measurement_denominator is None
        assert shape.measurement_denominator_unresolved is False


def _stat_vars(n: int) -> list[StatVar]:
    return [
        StatVar(
            ref=GraphRef(dcid=f"sv{i}", label=f"SV {i}"),
            shape_id="test-shape",
            slot_values=[],
            constraint_properties=[],
        )
        for i in range(n)
    ]


class TestResolvedStatVarsCap:
    """`_resolved_stat_vars` is the sole place resolved.stat_vars.items gets
    capped for the wire (see #659's production incident: an uncapped 1161-
    member scope shipped a 1.6 MiB response). Covers the cap's three
    contracts: a scope under cap publishes everything untruncated, a scope
    over cap publishes exactly cap items while member_count still reports
    the true full scope, and cap=None (the escape hatch internal-only
    callers such as options/resolve.py use) stays fully uncapped."""

    def test_scope_under_cap_publishes_every_item_untruncated(self):
        stat_vars = _stat_vars(5)

        resolved = _resolved_stat_vars(stat_vars, cap=10)

        assert [item.ref for item in resolved.items] == [sv.ref.dcid for sv in stat_vars]
        assert resolved.member_count == 5
        assert resolved.truncated is False

    def test_scope_over_cap_publishes_exactly_cap_with_full_member_count(self):
        stat_vars = _stat_vars(15)

        resolved = _resolved_stat_vars(stat_vars, cap=10)

        assert len(resolved.items) == 10
        assert [item.ref for item in resolved.items] == [sv.ref.dcid for sv in stat_vars[:10]]
        assert resolved.member_count == 15
        assert resolved.truncated is True

    def test_cap_none_is_uncapped_for_the_internal_only_escape_hatch(self):
        stat_vars = _stat_vars(500)

        resolved = _resolved_stat_vars(stat_vars, cap=None)

        assert len(resolved.items) == 500
        assert resolved.member_count == 500
        assert resolved.truncated is False

    def test_member_count_override_reports_the_true_scope_a_capped_union_understates(self):
        """same_shape.py's merge passes the sum of each group member's own
        member_count instead of len(stat_vars): a merged union built from
        already-capped per-member samples must not silently undercount the
        true population, and truncated must reflect that true population
        even when the union it was handed happens to sit at or under cap."""
        stat_vars = _stat_vars(8)

        resolved = _resolved_stat_vars(stat_vars, cap=10, member_count=1000)

        assert len(resolved.items) == 8
        assert resolved.member_count == 1000
        assert resolved.truncated is True


class TestTruncatedMeasuresAgainstMemberCount:
    """#893: truncated must measure `len(items) < member_count`, not
    `cap is not None and effective_count > cap` -- a cap-only signal that
    can never fire on an uncapped call (cap=None), the exact call shape
    standard_resolve's single-match branch feeds through region/pipeline.py
    once its own confirmed population exceeds 1. This is the assertion
    that would have caught #893's bug at this projection layer rather than
    only at the resolver."""

    def test_uncapped_call_with_explicit_member_count_still_reports_truncated(self):
        stat_vars = _stat_vars(1)

        resolved = _resolved_stat_vars(stat_vars, cap=None, member_count=12)

        assert len(resolved.items) == 1
        assert resolved.member_count == 12
        assert resolved.truncated is True


class TestBuildSpecStatVarsCap:
    """build_spec's own stat_vars_cap wiring end to end: the default caps a
    wire-facing Spec, and an internal-only caller (options/resolve.py's
    resolve_spec_resubmit call is the one production example) can opt out
    with stat_vars_cap=None and keep every resolved StatVar whole."""

    def _shape(self):
        shape_draft = _shape_draft(meas_denom_dcid=None, meas_denom_unresolved=False)
        return build_shape_model(
            shape_draft, slot_keys=[], five_tuple_refs={}, member_count=250, members=None
        )

    def test_default_cap_applies_to_a_wire_facing_spec(self):
        stat_vars = _stat_vars(2050)

        spec = build_spec(
            measure_elected_properties=frozenset(),
            shape=self._shape(),
            slots=[],
            stat_vars=stat_vars,
            entities=[],
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[],
        )

        assert len(spec.resolved.stat_vars.items) == 2000
        assert spec.resolved.stat_vars.member_count == 2050
        assert spec.resolved.stat_vars.truncated is True

    def test_stat_vars_cap_none_keeps_an_internal_consumer_whole(self):
        stat_vars = _stat_vars(250)

        spec = build_spec(
            measure_elected_properties=frozenset(),
            shape=self._shape(),
            slots=[],
            stat_vars=stat_vars,
            entities=[],
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[],
            stat_vars_cap=None,
        )

        assert len(spec.resolved.stat_vars.items) == 250
        assert spec.resolved.stat_vars.member_count == 250
        assert spec.resolved.stat_vars.truncated is False

    def test_the_trace_lists_the_first_200_stat_vars_and_says_so(self):
        stat_vars = _stat_vars(250)

        spec = build_spec(
            measure_elected_properties=frozenset(),
            shape=self._shape(),
            slots=[],
            stat_vars=stat_vars,
            entities=[],
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[],
            stat_vars_cap=None,
        )

        trace = spec.resolved.trace
        assert trace.resolved_stat_vars == [sv.ref for sv in stat_vars[:200]]
        assert trace.resolved_stat_vars_truncated is True

    def test_a_trace_under_200_stat_vars_lists_them_all(self):
        stat_vars = _stat_vars(5)

        spec = build_spec(
            measure_elected_properties=frozenset(),
            shape=self._shape(),
            slots=[],
            stat_vars=stat_vars,
            entities=[],
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[],
        )

        assert spec.resolved.trace.resolved_stat_vars == [sv.ref for sv in stat_vars]
        assert spec.resolved.trace.resolved_stat_vars_truncated is False


def test_a_slot_pairs_with_its_draft_by_property_and_a_slot_without_one_by_axis():
    """The where, when and source slots carry no property. Each takes the draft on its own
    axis, so a source binding leaves the where slot its place."""
    slot_keys = (
        SlotKeyDraft(axis="how", property_dcid="flowType", label="flow type"),
        SlotKeyDraft(axis="how", property_dcid="sector", label="sector"),
        SlotKeyDraft(axis="where", property_dcid=None, label="place"),
        SlotKeyDraft(axis="when", property_dcid=None, label="time window"),
        SlotKeyDraft(axis="source", property_dcid=None, label="data source"),
    )
    place = SlotBindingDraft(
        axis="where", property_dcid=None, kind="value", value_dcids=["country/IND"]
    )
    flow = SlotBindingDraft(
        axis="how", property_dcid="flowType", kind="value", value_dcids=["Grant"]
    )
    source = source_binding("provenance/Named")

    pairs = pair_slot_drafts(slot_keys, [place, flow, source])

    assert [key for key, _draft in pairs] == list(slot_keys)
    assert [draft for _key, draft in pairs] == [flow, None, place, None, source]


class TestStatedMemberDigest:
    """`resolved.shape.member_digest` is stated only for the family the code index resolves
    for the spec's identity."""

    def _shape(self, digest: str | None):
        return build_shape_model(
            _shape_draft(meas_denom_dcid=None, meas_denom_unresolved=False),
            slot_keys=[],
            five_tuple_refs={},
            member_count=1,
            members=None,
            member_digest=digest,
        )

    def test_the_digest_of_the_identitys_index_family_is_kept(self):
        _, identity = mint_identity(education_aid_identity())
        family = index_family(get_index(), identity_query(identity))
        assert family is not None and len(family.rows) == CRS_MEMBER_COUNT
        digest = member_digest(row.dcid for row in family.rows)

        assert _shape_stating_identity_family(self._shape(digest), identity).member_digest == digest

    def test_a_digest_over_another_population_is_dropped(self):
        _, identity = mint_identity(education_aid_identity())
        _, imputed = mint_identity(education_aid_identity(source=IMPUTED_PROVENANCE))
        family = index_family(get_index(), identity_query(identity))
        assert family is not None
        digest = member_digest(row.dcid for row in family.rows)
        subset = member_digest(row.dcid for row in family.rows[:5])

        assert _shape_stating_identity_family(self._shape(subset), identity).member_digest is None
        assert _shape_stating_identity_family(self._shape(digest), imputed).member_digest is None

    def test_an_identity_the_index_does_not_resolve_states_no_digest(self):
        document = education_aid_identity()
        document["shape"]["population_type"] = "EconomicActivity"
        _, identity = mint_identity(document)

        shape = self._shape(member_digest(["Amount_EconomicActivity_Nominal"]))

        assert _shape_stating_identity_family(shape, identity).member_digest is None


def _constraint_arcs(**constraints: str) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{p: {"nodes": [{"dcid": v}]} for p, v in constraints.items()},
    }


_FEMALE_URBAN_ARCS = {
    "Count_Person_Female_Urban": _constraint_arcs(
        placeOfResidenceClassification="Urban", gender="Female"
    ),
    "Count_Person_Female": _constraint_arcs(gender="Female"),
    "Count_Person": _constraint_arcs(),
}


def _gender_slots() -> list[Slot]:
    return [
        Slot(
            key=SlotKey(axis="how", property=GraphRef(dcid="gender", label="gender"), label="g"),
            binding=BindingValue(
                value=SlotValue(
                    ref=GraphRef(dcid="Female", label="Female"), value_kind="enum_value"
                )
            ),
            expansion=None,
        ),
        Slot(
            key=SlotKey(
                axis="how",
                property=GraphRef(
                    dcid="placeOfResidenceClassification", label="placeOfResidenceClassification"
                ),
                label="u",
            ),
            binding=BindingUnbound(),
            expansion=None,
        ),
    ]


class TestConstraintProperties:
    """Each StatVar lists every constraint property it asserts, bound or not."""

    def _built(self) -> dict[str, StatVar]:
        refs = [GraphRef(dcid=dcid, label=dcid) for dcid in _FEMALE_URBAN_ARCS]
        member_facts = {sv: ArcsMemberFacts(arcs) for sv, arcs in _FEMALE_URBAN_ARCS.items()}
        stat_vars = build_stat_vars(refs, "s", _gender_slots(), member_facts=member_facts)
        return {sv.ref.dcid: sv for sv in stat_vars}

    def test_a_member_lists_an_unbound_constraint_that_slot_values_omits(self):
        urban = self._built()["Count_Person_Female_Urban"]

        assert urban.constraint_properties == ["gender", "placeOfResidenceClassification"]
        assert [sv.key.property.dcid for sv in urban.slot_values if sv.key.property] == ["gender"]

    def test_the_root_lists_only_its_bound_constraint(self):
        assert self._built()["Count_Person_Female"].constraint_properties == ["gender"]

    def test_a_statvar_without_constraints_lists_none(self):
        assert self._built()["Count_Person"].constraint_properties == []

    @pytest.mark.parametrize(
        ("source", "dcid", "facts"),
        [
            pytest.param("member_facts", "Count_Person_Female_Urban", {}, id="no-member-facts"),
            pytest.param("arcs_by_sv", "Count_Person_Female_Urban", {}, id="no-arcs-by-sv"),
            pytest.param(
                "member_facts",
                "sv_empty",
                {"sv_empty": ArcsMemberFacts({})},
                id="empty-member-facts",
            ),
            pytest.param("arcs_by_sv", "sv_empty", {"sv_empty": {}}, id="empty-arcs-by-sv"),
        ],
    )
    def test_a_statvar_with_no_facts_lists_null_rather_than_reading_as_the_root(
        self, source, dcid, facts
    ):
        """An empty list would match a root's value-bound slots, so unknown constraints are null."""
        refs = [GraphRef(dcid=dcid, label="urban")]

        (stat_var,) = build_stat_vars(refs, "s", _gender_slots(), **{source: facts})

        assert stat_var.constraint_properties is None
        assert stat_var.slot_values == []
        item = _resolved_stat_vars([stat_var], cap=None).items[0]
        assert item.model_dump(mode="json")["constraint_properties"] is None

    def test_the_wire_projection_and_its_rebuild_carry_the_list(self):
        urban = self._built()["Count_Person_Female_Urban"]

        item = _resolved_stat_vars([urban], cap=None).items[0]

        assert item.constraint_properties == urban.constraint_properties
        assert item.model_dump(mode="json")["constraint_properties"] == [
            "gender",
            "placeOfResidenceClassification",
        ]
        assert stat_var_from_resolved(item).constraint_properties == urban.constraint_properties

    def test_the_published_schema_requires_and_describes_the_field(self):
        schema = ResolvedStatVar.model_json_schema(mode="serialization")

        assert "constraint_properties" in schema["required"]
        field = schema["properties"]["constraint_properties"]
        assert field["anyOf"] == [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]
        assert "null means its constraints are not known" in field["description"]


class TestBuildSpecEntityGroups:
    """A role whose entities are exactly a named group's members carries the group on the
    definition and on resolved, outside the spec_id hash."""

    _BALTIC = NamedGroup(
        label="Baltic states",
        dcid=None,
        member_dcids=frozenset({"country/EST", "country/LTU", "country/LVA"}),
    )

    def _spec(self, origin_dcids: list[str], *, named_groups: tuple[NamedGroup, ...]):
        origin = EntityRoleDirectional(
            role=GraphRef(dcid="financingProvider", label="donor"), axis="origin"
        )
        destination = EntityRoleDirectional(
            role=GraphRef(dcid="financingRecipient", label="recipient"), axis="destination"
        )
        entities = [Entity(ref=GraphRef(dcid=d, label=d), role=origin) for d in origin_dcids]
        entities.append(
            Entity(ref=GraphRef(dcid="country/ETH", label="Ethiopia"), role=destination)
        )
        return build_spec(
            measure_elected_properties=frozenset(),
            shape=build_shape_model(
                _shape_draft(meas_denom_dcid=None, meas_denom_unresolved=False),
                slot_keys=[],
                five_tuple_refs={},
                member_count=1,
                members=None,
            ),
            slots=[],
            stat_vars=_stat_vars(1),
            entities=entities,
            coverage=CoverageBare(has_data=True),
            pipeline_trace=[],
            named_groups=named_groups,
        )

    def test_a_grouped_role_mints_the_ungrouped_spec_id_and_states_its_group(self):
        members = ["country/LVA", "country/EST", "country/LTU"]

        grouped = self._spec(members, named_groups=(self._BALTIC,))
        ungrouped = self._spec(members, named_groups=())

        expected = {"origin": IdentityEntityGroup(dcid=None, label="Baltic states")}
        assert grouped.spec_id == ungrouped.spec_id
        assert grouped.definition.identity.entity_groups == expected
        assert grouped.resolved.entity_groups == expected
        assert ungrouped.resolved.entity_groups is None

    @pytest.mark.parametrize(
        "origin_dcids",
        [
            pytest.param(["country/EST", "country/LVA"], id="fewer_members"),
            pytest.param(
                ["country/EST", "country/LTU", "country/LVA", "country/FIN"], id="extra_entity"
            ),
        ],
    )
    def test_a_role_that_is_not_exactly_the_members_carries_no_group(self, origin_dcids):
        spec = self._spec(origin_dcids, named_groups=(self._BALTIC,))

        assert spec.definition.identity.entity_groups is None
        assert spec.resolved.entity_groups is None
