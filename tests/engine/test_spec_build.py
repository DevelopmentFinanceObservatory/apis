"""Tests denominator refs and unresolved flags in the wire `Shape`."""

from __future__ import annotations

from dataclasses import replace

import pytest

from code_index.members import antichain_minimal_rows, index_family
from code_index.shared import get_index
from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.bound_values import (
    StatedMembers,
    WalkedMembers,
    identity_query,
    index_decides_membership,
    source_binding,
    stated_members,
    walk_key,
)
from search.engine.retrieval.member_facts import ArcsMemberFacts
from search.engine.shape import ShapeDraft, SlotKeyDraft
from search.engine.spec_build import (
    NamedGroup,
    build_shape_model,
    build_spec,
    build_stat_vars,
    pair_slot_drafts,
    spec_named_groups,
)
from search.engine.spec_id import mint_identity
from search.models import (
    BindingUnbound,
    BindingValue,
    CoverageBare,
    DateRange,
    Entity,
    EntityGroup,
    EntityRoleDirectional,
    GraphRef,
    MembersNotStated,
    Slot,
    SlotKey,
    SlotValue,
    Spec,
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


class TestBuildSpecStatVars:
    """`build_spec` holds every served member off the wire and states their counts and the
    dates they serve."""

    def _shape(self):
        shape_draft = _shape_draft(meas_denom_dcid=None, meas_denom_unresolved=False)
        return build_shape_model(
            shape_draft, slot_keys=[], five_tuple_refs={}, member_count=250, members=None
        )

    def _build(self, stat_vars: list[StatVar], **kwargs) -> Spec:
        return build_spec(
            open_roles=(),
            measure_elected_properties=frozenset(),
            shape=self._shape(),
            slots=[],
            stat_vars=stat_vars,
            entities=[],
            coverage=CoverageBare(has_data=True),
            **kwargs,
        )

    def test_every_member_is_held_and_counted(self):
        stat_vars = _stat_vars(2050)

        spec = self._build(stat_vars)

        assert spec.resolved.members == tuple(stat_vars)
        assert len(spec.resolved.sentence_facts.members) == 2050
        assert spec.resolved.stat_vars.member_count == 2050

    def test_a_spec_rebuilt_from_its_wire_form_refuses_its_members_and_facts(self):
        spec = self._build(_stat_vars(3))

        rebuilt = Spec.model_validate(spec.model_dump())

        assert rebuilt == spec
        with pytest.raises(MembersNotStated):
            rebuilt.resolved.members  # noqa: B018 -- the read raises
        with pytest.raises(MembersNotStated):
            rebuilt.resolved.sentence_facts  # noqa: B018 -- the read raises

    @pytest.mark.parametrize(
        ("ranges", "served"),
        [
            pytest.param(
                [DateRange(start="2015", end="2020"), DateRange(start="2010", end="2018"), None],
                DateRange(start="2010", end="2020"),
                id="earliest-start-latest-end",
            ),
            pytest.param([DateRange(start="2015"), None], DateRange(start="2015"), id="start-only"),
            pytest.param([None, None], None, id="none-served"),
        ],
    )
    def test_the_served_date_range_spans_every_members_range(self, ranges, served):
        stat_vars = [
            stat_var.model_copy(update={"data_date_range": date_range})
            for stat_var, date_range in zip(_stat_vars(len(ranges)), ranges, strict=True)
        ]

        assert self._build(stat_vars).resolved.stat_vars.data_date_range == served

    def test_a_walk_of_the_identitys_own_query_states_the_served_count_and_digest(self):
        unwalked = self._build(_stat_vars(3))
        identity = unwalked.definition.identity
        assert not index_decides_membership(identity)
        stated = StatedMembers(served_count=2, digest=member_digest(["sv0", "sv1", "sv2"]))
        own = WalkedMembers(key=walk_key(identity_query(identity)), stated=stated)
        other_key = (*own.key[:2], frozenset({"origin"}), own.key[3])
        open_key = (*own.key[:3], frozenset({"origin"}))

        walked = self._build(_stat_vars(3), walked=own)
        other = self._build(_stat_vars(3), walked=replace(own, key=other_key))
        opened = self._build(_stat_vars(3), walked=replace(own, key=open_key))

        assert (
            unwalked.resolved.stat_vars.served_count,
            unwalked.resolved.shape.member_digest,
        ) == (
            None,
            None,
        )
        assert (walked.resolved.stat_vars.served_count, walked.resolved.shape.member_digest) == (
            2,
            stated.digest,
        )
        assert (other.resolved.stat_vars.served_count, other.resolved.shape.member_digest) == (
            None,
            None,
        )
        assert (opened.resolved.stat_vars.served_count, opened.resolved.shape.member_digest) == (
            None,
            None,
        )


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


class TestStatedMembers:
    """What a Spec states of the mint's members (`stated_members`): the family the code index
    resolves for its identity, else the one the walk of its own query settled."""

    def test_an_index_family_states_its_antichain_count_and_every_rows_digest(self):
        _, identity = mint_identity(education_aid_identity())
        family = index_family(get_index(), identity_query(identity))
        assert family is not None and len(family.rows) == CRS_MEMBER_COUNT

        assert stated_members(identity, None) == StatedMembers(
            served_count=len(antichain_minimal_rows(get_index(), family.rows)),
            digest=member_digest(row.dcid for row in family.rows),
        )

    def test_an_identity_the_index_resolves_no_family_for_states_none(self):
        """The index resolves no family for the education-aid identity naming the imputed
        dataset, so Search states none."""
        _, imputed = mint_identity(education_aid_identity(source=IMPUTED_PROVENANCE))

        assert stated_members(imputed, None) is None

    def test_an_identity_the_index_does_not_resolve_states_only_its_own_walk(self):
        document = education_aid_identity()
        document["shape"]["population_type"] = "EconomicActivity"
        _, identity = mint_identity(document)
        stated = StatedMembers(served_count=1, digest=member_digest(["Amount_EconomicActivity"]))
        walked = WalkedMembers(key=walk_key(identity_query(identity)), stated=stated)
        _, other_identity = mint_identity(education_aid_identity())
        elsewhere = replace(walked, key=walk_key(identity_query(other_identity)))

        assert stated_members(identity, None) is None
        assert stated_members(identity, walked) == stated
        assert stated_members(identity, elsewhere) is None


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


class TestBuildSpecEntityGroups:
    """A role whose entities are exactly a named group's members serves the group on
    `resolved.entity_groups`."""

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
            open_roles=(),
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
            named_groups=named_groups,
        )

    @pytest.mark.parametrize("reverse", [False, True], ids=["as-listed", "reversed"])
    @pytest.mark.parametrize(
        "other_dcid", ["undata-geo/G99999999", None], ids=["other-has-a-node", "other-has-none"]
    )
    def test_of_groups_with_one_member_list_a_role_carries_the_first_by_dcid(
        self, reverse, other_dcid
    ):
        """Of groups with the same members, one with a node wins over one without, then
        the first by dcid, in either order."""
        baltic = replace(self._BALTIC, dcid="undata-geo/G00000000")
        other = replace(self._BALTIC, dcid=other_dcid, label="Baltic rim")
        groups = (other, baltic) if reverse else (baltic, other)

        spec = self._spec(sorted(baltic.member_dcids), named_groups=groups)

        assert spec.resolved.entity_groups == {
            "origin": EntityGroup(dcid=baltic.dcid, label=baltic.label)
        }

    def test_a_grouped_role_mints_the_ungrouped_spec_id_and_states_its_group(self):
        members = ["country/LVA", "country/EST", "country/LTU"]

        grouped = self._spec(members, named_groups=(self._BALTIC,))
        ungrouped = self._spec(members, named_groups=())

        expected = {"origin": EntityGroup(dcid=None, label="Baltic states")}
        assert grouped.spec_id == ungrouped.spec_id
        assert grouped.definition.identity == ungrouped.definition.identity
        assert grouped.resolved.entity_groups == expected
        assert ungrouped.resolved.entity_groups is None

    @pytest.mark.parametrize("entity_type", ["UNGeoRegion", None])
    def test_a_group_serves_the_type_of_its_node(self, entity_type):
        baltic = replace(self._BALTIC, dcid="undata-geo/G00000000", entity_type=entity_type)

        spec = self._spec(sorted(baltic.member_dcids), named_groups=(baltic,))

        assert spec.resolved.entity_groups == {
            "origin": EntityGroup(
                dcid="undata-geo/G00000000", label="Baltic states", entity_type=entity_type
            )
        }
        assert spec_named_groups(spec) == (baltic,)
        assert spec.spec_id == self._spec(sorted(baltic.member_dcids), named_groups=()).spec_id

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

        assert spec.resolved.entity_groups is None

    def test_a_role_names_each_set_it_binds_whole(self):
        """A role binding a group's members beside another place carries no group and
        names the set, which `spec_named_groups` returns."""
        whole = [*sorted(self._BALTIC.member_dcids), "country/FIN"]

        beside = self._spec(whole, named_groups=(self._BALTIC,))
        partial = self._spec(whole[1:], named_groups=(self._BALTIC,))

        assert beside.resolved.entity_groups is None
        assert beside.resolved.named_sets == (
            ("origin", EntityGroup(dcid=None, label="Baltic states"), self._BALTIC.member_dcids),
        )
        assert spec_named_groups(beside) == (self._BALTIC,)
        assert partial.resolved.named_sets == ()

    def test_a_role_does_not_name_a_set_it_binds_at_its_node(self):
        """A role binding a set's node beside every member of the set serves the set at
        its node. The members came from other names, so the set is none of its named
        sets."""
        baltic = replace(self._BALTIC, dcid="undata-geo/G00000000")

        spec = self._spec([*sorted(baltic.member_dcids), baltic.dcid], named_groups=(baltic,))

        assert spec.resolved.named_sets == ()
