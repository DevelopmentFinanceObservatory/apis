"""Tests alternate-shape representatives and the elected-data gate.

Alternate shapes have no real bind call, so their representative must agree with elected values or
remain unqualified.
"""

from __future__ import annotations

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.representative import (
    shape_agreement_representative,
    shape_bare_representative,
)
from search.engine.retrieval.retrieve import Materialised
from search.engine.shape import ShapeDraft, SlotKeyDraft
from tests.engine._builders import (
    bound_place_context,
    materialise_alternates,
    subject_place_roles,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph


def _arcs(
    *,
    pop_type: str = "TestPopType",
    meas_prop: str = "testMeasure",
    stat_type: str = "measuredValue",
    **constraints: str,
) -> dict:
    """Build five-tuple and constraint arcs for a StatVar fixture."""
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": pop_type}]},
        "measuredProperty": {"nodes": [{"dcid": meas_prop}]},
        "statType": {"nodes": [{"dcid": stat_type}]},
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
    }
    for prop, value in constraints.items():
        arcs[prop] = {"nodes": [{"dcid": value}]}
    return arcs


def _arcs_with_unresolved(
    prop: str,
    *,
    pop_type: str = "TestPopType",
    meas_prop: str = "testMeasure",
    stat_type: str = "measuredValue",
    **constraints: str,
) -> dict:
    """Build `_arcs` with a dcid-less ProvisionalNode target for `prop`."""
    arcs = _arcs(pop_type=pop_type, meas_prop=meas_prop, stat_type=stat_type, **constraints)
    arcs["constraintProperties"]["nodes"].append({"dcid": prop})
    arcs[prop] = {"nodes": [{"types": ["ProvisionalNode"], "provenanceId": "dc/base/Schema"}]}
    return arcs


def _shape(
    *,
    members: dict[str, dict],
    slot_keys: tuple[SlotKeyDraft, ...] = (),
    slot_taxonomy: dict[str, list[str]] | None = None,
) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_shape",
        label="test shape",
        pop_type_dcid="TestPopType",
        meas_prop_dcid="testMeasure",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=slot_keys,
        sv_arc_facts=members,
        slot_taxonomy=slot_taxonomy or {},
        representative_score=0.9,
    )


_GENDER_SLOT = SlotKeyDraft(axis="how", property_dcid="gender", label="gender")
_AGE_SLOT = SlotKeyDraft(axis="how", property_dcid="age", label="age")
_HCF_SLOT = SlotKeyDraft(axis="how", property_dcid="healthCareFunction", label="health function")
_PORC = "placeOfResidenceClassification"
_PORC_SLOT = SlotKeyDraft(axis="how", property_dcid=_PORC, label="residence")


class TestShapeBareRepresentative:
    """A constraint is a disagreement axis only when members show at least two values."""

    def test_single_member_shape_returns_its_only_member(self):
        shape = _shape(
            members={"sv_female": _arcs(gender="Female")},
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female"]},
        )
        assert shape_bare_representative(shape) == "sv_female"

    def test_all_members_sharing_one_value_is_not_a_disagreement_axis(self):
        """Two members both assert gender=Female (e.g. two disaggregations of a
        Female-only series) -- taxonomy collapses to one observed value, so
        gender is a fixed fact, not a real choice; the first member wins."""
        shape = _shape(
            members={
                "sv_female_a": _arcs(gender="Female"),
                "sv_female_b": _arcs(gender="Female"),
            },
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female"]},
        )
        assert shape_bare_representative(shape) == "sv_female_a"

    def test_finds_the_bare_member_among_a_real_disagreement_axis(self):
        """gender has 2 observed values (a real disagreement axis); only the
        member silent on gender honestly represents "unqualified" -- found
        regardless of insertion order (inserted last here, not first)."""
        shape = _shape(
            members={
                "sv_female": _arcs(gender="Female"),
                "sv_male": _arcs(gender="Male"),
                "sv_bare": _arcs(),
            },
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female", "Male"]},
        )
        assert shape_bare_representative(shape) == "sv_bare"

    def test_returns_none_when_every_member_takes_a_stance(self):
        """gender has 2 observed values and no member is silent on it -- no
        honest "unqualified" reading exists (nn-04/std-07's SDG shape)."""
        shape = _shape(
            members={
                "sv_female": _arcs(gender="Female"),
                "sv_male": _arcs(gender="Male"),
            },
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female", "Male"]},
        )
        assert shape_bare_representative(shape) is None

    def test_returns_none_for_a_shape_with_no_confirmed_members(self):
        shape = _shape(members={}, slot_keys=(_GENDER_SLOT,))
        assert shape_bare_representative(shape) is None


class TestShapeBareRepresentativeUnresolvedAxis:
    """An unresolved arc excludes a member from the bare representative, even without a dcid."""

    def test_present_but_unresolved_member_is_not_picked_as_bare(self):
        shape = _shape(
            members={
                "sv_female": _arcs(gender="Female"),
                "sv_male": _arcs(gender="Male"),
                "sv_unresolved": _arcs_with_unresolved("gender"),
            },
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female", "Male"]},
        )
        assert shape_bare_representative(shape) is None

    def test_genuinely_bare_member_still_wins_alongside_an_unresolved_one(self):
        shape = _shape(
            members={
                "sv_female": _arcs(gender="Female"),
                "sv_male": _arcs(gender="Male"),
                "sv_unresolved": _arcs_with_unresolved("gender"),
                "sv_bare": _arcs(),
            },
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female", "Male"]},
        )
        assert shape_bare_representative(shape) == "sv_bare"


class TestShapeAgreementRepresentative:
    """Prefer agreement with elected values before the bare representative."""

    @pytest.mark.parametrize(
        ("members", "slot_keys", "slot_taxonomy", "elected_values", "expected"),
        [
            pytest.param(
                {
                    "sv_bare": _arcs(),
                    "sv_hc1": _arcs(healthCareFunction="HC_1"),
                    "sv_hc11": _arcs(healthCareFunction="HC_11"),
                },
                (_HCF_SLOT,),
                {"how:healthCareFunction": ["HC_1", "HC_11"]},
                {"healthCareFunction": ["HC_1"]},
                "sv_hc1",
                id="agreement_precedes_bare_member",
            ),
            pytest.param(
                {
                    "hc1_urban": _arcs(healthCareFunction="HC_1", **{_PORC: "Urban"}),
                    "hc11_urban": _arcs(healthCareFunction="HC_11", **{_PORC: "Urban"}),
                    "hc11_rural": _arcs(healthCareFunction="HC_11", **{_PORC: "Rural"}),
                    "hc1_bare_residence": _arcs(healthCareFunction="HC_1"),
                },
                (_HCF_SLOT, _PORC_SLOT),
                {
                    "how:healthCareFunction": ["HC_1", "HC_11"],
                    f"how:{_PORC}": ["Urban", "Rural"],
                },
                {"healthCareFunction": ["HC_1"]},
                "hc1_bare_residence",
                id="two_axis_agreement_stays_silent",
            ),
            pytest.param(
                {
                    "hc1_unresolved": _arcs_with_unresolved("healthCareFunction"),
                    "hc1_bare": _arcs(healthCareFunction="HC_1"),
                    "hc11_other": _arcs(healthCareFunction="HC_11"),
                },
                (_HCF_SLOT,),
                {"how:healthCareFunction": ["HC_1", "HC_11"]},
                {"healthCareFunction": ["HC_1"]},
                "hc1_bare",
                id="unresolved_member_skipped_in_agreement",
            ),
            pytest.param(
                {
                    "sv_hc1": _arcs(healthCareFunction="HC_1"),
                    "sv_hc11": _arcs(healthCareFunction="HC_11"),
                    "sv_unresolved": _arcs_with_unresolved("healthCareFunction"),
                    "sv_bare": _arcs(),
                },
                (_HCF_SLOT,),
                {"how:healthCareFunction": ["HC_1", "HC_11"]},
                {"gender": ["Female"]},
                "sv_bare",
                id="unresolved_member_skipped_in_bare_fallback",
            ),
        ],
    )
    def test_representative_selection(
        self, members, slot_keys, slot_taxonomy, elected_values, expected
    ):
        shape = _shape(
            members=members,
            slot_keys=slot_keys,
            slot_taxonomy=slot_taxonomy,
        )
        assert shape_agreement_representative(shape, elected_values) == expected

    def test_two_axis_regression_agreement_on_one_axis_must_not_smuggle_another(self):
        """An HC_1 member that also asserts Urban cannot represent an unqualified alternative."""
        shape = _shape(
            members={
                "hc1_urban": _arcs(healthCareFunction="HC_1", **{_PORC: "Urban"}),
                "hc1_rural": _arcs(healthCareFunction="HC_1", **{_PORC: "Rural"}),
                "hc11_urban": _arcs(healthCareFunction="HC_11", **{_PORC: "Urban"}),
            },
            slot_keys=(_HCF_SLOT, _PORC_SLOT),
            slot_taxonomy={
                "how:healthCareFunction": ["HC_1", "HC_11"],
                f"how:{_PORC}": ["Urban", "Rural"],
            },
        )
        rep = shape_agreement_representative(shape, {"healthCareFunction": ["HC_1"]})
        assert rep is None

    def test_falls_back_to_bare_when_no_member_shares_the_elected_property(self):
        shape = _shape(
            members={
                "sv_bare": _arcs(),
                "sv_urban": _arcs(placeOfResidenceClassification="Urban"),
            },
            slot_keys=(
                SlotKeyDraft(
                    axis="how",
                    property_dcid="placeOfResidenceClassification",
                    label="residence",
                ),
            ),
            slot_taxonomy={"how:placeOfResidenceClassification": ["Urban", "Rural"]},
        )
        rep = shape_agreement_representative(shape, {"gender": ["Female"]})
        assert rep == "sv_bare"

    def test_value_mismatch_on_a_shared_property_falls_through_to_none(self):
        """A shared property with incompatible value spaces does not count as agreement."""
        shape = _shape(
            members={
                "sdg_sex_f": _arcs(age="SDG_AgeEnum_Y0", sdg_sex="SDG_SexEnum_F"),
                "sdg_sex_m": _arcs(age="SDG_AgeEnum_Y0", sdg_sex="SDG_SexEnum_M"),
            },
            slot_keys=(
                _AGE_SLOT,
                SlotKeyDraft(axis="how", property_dcid="sdg_sex", label="sex"),
            ),
            slot_taxonomy={
                "how:age": ["SDG_AgeEnum_Y0"],
                "how:sdg_sex": ["SDG_SexEnum_F", "SDG_SexEnum_M"],
            },
        )
        rep = shape_agreement_representative(shape, {"age": ["Years0"]})
        assert rep is None

    def test_no_elected_bound_values_falls_back_to_bare_directly(self):
        shape = _shape(
            members={
                "sv_female": _arcs(gender="Female"),
                "sv_bare": _arcs(),
            },
            slot_keys=(_GENDER_SLOT,),
            slot_taxonomy={"how:gender": ["Female"]},
        )
        assert shape_agreement_representative(shape, {}) == "sv_female"


class TestMaterialiseAltShapesElectedHasDataGate:
    """Drop shapes without an honest unqualified reading when the elected shape has data.

    Keep their real data as `alternate_match` when the elected shape has none.
    """

    _RECIPIENT = "country/USA"
    _SV_URBAN = "sv_urban"
    _SV_RURAL = "sv_rural"

    def _no_bare_reading_shape(self) -> ShapeDraft:
        return _shape(
            members={
                self._SV_URBAN: _arcs(placeOfResidenceClassification="Urban"),
                self._SV_RURAL: _arcs(placeOfResidenceClassification="Rural"),
            },
            slot_keys=(
                SlotKeyDraft(
                    axis="how",
                    property_dcid="placeOfResidenceClassification",
                    label="residence",
                ),
            ),
            slot_taxonomy={"how:placeOfResidenceClassification": ["Urban", "Rural"]},
        )

    def _graph_with_urban_data(self) -> FakeGraph:
        obs = {
            f"{self._SV_URBAN}|{self._RECIPIENT}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        }
        return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})

    def test_elected_has_data_true_drops_the_shape_before_probing(self):
        results, unsupported = materialise_alternates(
            [self._no_bare_reading_shape()],
            places=[bound_place_context(subject_place_roles(self._RECIPIENT))],
            elected_bindings=[],
            elected_has_data=True,
            named_place_dcids=[self._RECIPIENT],
            graph=self._graph_with_urban_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            date_request=None,
            coverage=empty_coverage_snapshot(),
        )
        assert unsupported is None
        assert results == []

    def test_elected_has_data_false_still_surfaces_real_data(self):
        results, unsupported = materialise_alternates(
            [self._no_bare_reading_shape()],
            places=[bound_place_context(subject_place_roles(self._RECIPIENT))],
            elected_bindings=[],
            elected_has_data=False,
            named_place_dcids=[self._RECIPIENT],
            graph=self._graph_with_urban_data(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            date_request=None,
            coverage=empty_coverage_snapshot(),
        )
        assert unsupported is None
        assert len(results) == 1
        bindings, result = results[0].bindings, results[0].mat
        assert isinstance(result, Materialised)
        assert result.has_data is True
        assert result.sv_dcids == [self._SV_URBAN]
        assert any(
            b.property_dcid == "placeOfResidenceClassification" and b.value_dcids == ["Urban"]
            for b in bindings
        )

    def test_agreement_survives_regardless_of_elected_has_data(self):
        """A shape that DOES have an honest reading (agreement or bare) is
        never dropped by the gate itself -- elected_has_data only controls
        whether the agreement/bare check runs at all, not a second filter on
        top of it."""
        who04b_shape = _shape(
            members={
                "sv_hc1_pc": _arcs(healthCareFunction="HC_1"),
                "sv_hc11_pc": _arcs(healthCareFunction="HC_11"),
            },
            slot_keys=(_HCF_SLOT,),
            slot_taxonomy={"how:healthCareFunction": ["HC_1", "HC_11"]},
        )
        obs = {
            f"sv_hc1_pc|{self._RECIPIENT}": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        }
        graph = FakeGraph(nodes={}, obs=obs, detect={}, resolve={})
        elected_bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="healthCareFunction",
                kind="value",
                value_dcids=["HC_1"],
            ),
        ]

        results, unsupported = materialise_alternates(
            [who04b_shape],
            places=[bound_place_context(subject_place_roles(self._RECIPIENT))],
            elected_bindings=elected_bindings,
            elected_has_data=True,
            named_place_dcids=[self._RECIPIENT],
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            date_request=None,
            coverage=empty_coverage_snapshot(),
        )
        assert unsupported is None
        assert len(results) == 1
        assert results[0].mat.sv_dcids == ["sv_hc1_pc"]
