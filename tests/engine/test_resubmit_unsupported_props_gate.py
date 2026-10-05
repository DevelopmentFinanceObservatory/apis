"""Test that resubmit narrowing preserves only members supporting untouched bindings.

A ratio-backed narrow answers from a fully asserting survivor, since construct-and-confirm
returns nothing for ratio families. Otherwise construct-and-confirm seeds from the survivors,
not the posted set. An edited set keeps only values a final survivor asserts.
"""

from __future__ import annotations

from search.engine.region import resolve_spec_resubmit
from search.models import (
    BindingValue,
    ResubmitIdentityPayload,
    Slot,
    SlotValue,
)
from tests.engine._base_identity import (
    PERSON_COUNT_SHAPE,
    make_base,
    make_identity,
    subject_entity,
)
from tests.engine._builders import (
    binding_set_slot as _set_slot,
)
from tests.engine._builders import (
    gref as _gref,
)
from tests.engine._builders import (
    value_slot as _value_slot,
)
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

# Ratio-backed narrow selects a survivor that asserts every retained binding.

_KEN_DCID = "country/KEN"

# Mirrors _mortality_sv_node's own populationType/measuredProperty/statType/
# measurementDenominator arcs below -- resolve_spec_resubmit's standard path
# re-derives the shape from the posted resolved_stat_var_dcids' real arcs and
# requires it to match the posted shape structurally.
_RATIO_SHAPE: dict = {
    "kind": "statvar_schema",
    "population_type": "MortalityEvent",
    "measured_property": "count",
    "stat_type": "measuredValue",
    "measurement_denominator": {"ref": "Count_BirthEvent_LiveBirth"},
}
_RATIO_YOUTH_SV = "Count_MortalityEvent_Youth_AsAFractionOf_Count_BirthEvent_LiveBirth"
_RATIO_INFANT_SILENT_SV = "Count_MortalityEvent_Infant_AsAFractionOf_Count_BirthEvent_LiveBirth"
_RATIO_INFANT_FULL_SV = (
    "Count_MortalityEvent_Infant_Female_AsAFractionOf_Count_BirthEvent_LiveBirth"
)


def _mortality_sv_node(label: str, age_group: str, gender: str | None = None) -> dict:
    """Build a ratio-backed MortalityEvent StatVar with the shared fixture shape."""
    constraint_props = [{"dcid": "ageGroupClassification"}]
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "MortalityEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementDenominator": {"nodes": [{"dcid": "Count_BirthEvent_LiveBirth"}]},
        "ageGroupClassification": {"nodes": [{"dcid": age_group}]},
    }
    if gender is not None:
        constraint_props.append({"dcid": "gender"})
        arcs["gender"] = {"nodes": [{"dcid": gender}]}
    arcs["constraintProperties"] = {"nodes": constraint_props}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


def _ratio_gate_graph() -> FakeGraph:
    """Three ratio siblings share a shape. Infant(full) asserts age and gender."""
    nodes = {
        _KEN_DCID: {"label": "Kenya", "type": "Country"},
        "Youth": {"label": "Youth", "type": None},
        "Infant": {"label": "Infant", "type": None},
        "Female": {"label": "Female", "type": None},
        _RATIO_YOUTH_SV: _mortality_sv_node("Youth mortality rate", "Youth"),
        _RATIO_INFANT_SILENT_SV: _mortality_sv_node("Infant mortality rate", "Infant"),
        _RATIO_INFANT_FULL_SV: _mortality_sv_node(
            "Infant female mortality rate", "Infant", gender="Female"
        ),
    }
    obs_entry = [
        {"earliestDate": "2010", "latestDate": "2020", "obsCount": 3, "provenanceId": "dc/base/t"}
    ]
    obs = {f"{_RATIO_INFANT_FULL_SV}|{_KEN_DCID}": obs_entry}
    return FakeGraph(nodes=nodes, obs=obs)


class TestRatioBackedNarrowAnswersFromFullAssertingSurvivor:
    def test_narrow_resolves_definite_from_full_asserter_not_no_data(self):
        entities = [subject_entity(_KEN_DCID)]
        age_group_set = _set_slot("how", "ageGroupClassification", "age group", ["Youth", "Infant"])
        base_spec_id, base_identity = make_base(
            _RATIO_SHAPE, slots=[age_group_set], entities=entities
        )

        # Narrow to Infant. Only the full sibling asserts the untouched Female binding.
        age_group_narrowed = Slot(
            key=age_group_set.key,
            binding=BindingValue(value=SlotValue(ref=_gref("Infant"), value_kind="enum_value")),
        )
        gender_slot = _value_slot("how", "gender", "gender", "Female")
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                _RATIO_SHAPE, [age_group_narrowed, gender_slot], entities=entities
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[
                _RATIO_YOUTH_SV,
                _RATIO_INFANT_SILENT_SV,
                _RATIO_INFANT_FULL_SV,
            ],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_ratio_gate_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        resolved_svs = [sv.ref for sv in spec.resolved.stat_vars.items]
        assert resolved_svs == [_RATIO_INFANT_FULL_SV]

        gender_result = next(
            s for s in spec.resolved.slots if s.key.property and s.key.property.dcid == "gender"
        )
        assert isinstance(gender_result.binding, BindingValue)
        assert gender_result.binding.value.ref is not None
        assert gender_result.binding.value.ref.dcid == "Female"


# Construct-and-confirm uses the narrow's survivors as its seed.

_IND_DCID = "country/IND"

_URBAN_MALE_SV = "Count_Person_Urban_Male"
_RURAL_FEMALE_SV = "Count_Person_Rural_Female"
_RURAL_FEMALE_ADULT_SV = "Count_Person_Rural_Female_Years18Onwards"
# This graph-only sibling detects construct-and-confirm seeded from all posted members.
_LEAK_RURAL_MALE_ADULT_SV = "Count_Person_Rural_Male_Years18Onwards"


def _person_sv_node(label: str, residence: str, gender: str, age: str | None = None) -> dict:
    constraint_props = [{"dcid": "gender"}, {"dcid": "placeOfResidenceClassification"}]
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "gender": {"nodes": [{"dcid": gender}]},
        "placeOfResidenceClassification": {"nodes": [{"dcid": residence}]},
    }
    if age is not None:
        constraint_props.append({"dcid": "age"})
        arcs["age"] = {"nodes": [{"dcid": age}]}
    arcs["constraintProperties"] = {"nodes": constraint_props}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


def _leak_graph() -> FakeGraph:
    """A graph-only sibling detects seeding from all candidates."""
    nodes = {
        _IND_DCID: {"label": "India", "type": "Country"},
        "Urban": {"label": "Urban", "type": None},
        "Rural": {"label": "Rural", "type": None},
        "Male": {"label": "Male", "type": None},
        "Female": {"label": "Female", "type": None},
        "Years18Onwards": {"label": "Years18Onwards", "type": None},
        _URBAN_MALE_SV: _person_sv_node("Urban male population", "Urban", "Male"),
        _RURAL_FEMALE_SV: _person_sv_node("Rural female population", "Rural", "Female"),
        _RURAL_FEMALE_ADULT_SV: _person_sv_node(
            "Adult rural female population", "Rural", "Female", age="Years18Onwards"
        ),
        _LEAK_RURAL_MALE_ADULT_SV: _person_sv_node(
            "Adult rural male population", "Rural", "Male", age="Years18Onwards"
        ),
    }
    obs_entry = [
        {"earliestDate": "2001", "latestDate": "2021", "obsCount": 3, "provenanceId": "dc/base/t"}
    ]
    obs = {
        f"{_RURAL_FEMALE_ADULT_SV}|{_IND_DCID}": obs_entry,
        # This impostor has data so an incorrect confirmation could pass availability checks.
        f"{_LEAK_RURAL_MALE_ADULT_SV}|{_IND_DCID}": obs_entry,
    }
    return FakeGraph(nodes=nodes, obs=obs)


class TestNarrowSeedsRefineFromSurvivorsNotOriginalPostedSet:
    def test_excluded_sibling_value_never_leaks_into_served_answer_or_identity(self):
        entities = [subject_entity(_IND_DCID)]
        residence_set_slot = _set_slot(
            "how", "placeOfResidenceClassification", "place of residence", ["Urban", "Rural"]
        )
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE, slots=[residence_set_slot], entities=entities
        )

        # Narrow residence to Rural. Keep gender and age bound.
        residence_narrowed_slot = Slot(
            key=residence_set_slot.key,
            binding=BindingValue(value=SlotValue(ref=_gref("Rural"), value_kind="enum_value")),
        )
        gender_set_slot = _set_slot("how", "gender", "gender", ["Male", "Female"])
        age_value_slot = _value_slot("how", "age", "age", "Years18Onwards")

        inp = ResubmitIdentityPayload(
            identity=make_identity(
                PERSON_COUNT_SHAPE,
                [residence_narrowed_slot, gender_set_slot, age_value_slot],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_URBAN_MALE_SV, _RURAL_FEMALE_SV, _RURAL_FEMALE_ADULT_SV],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_leak_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        # The age-bound response must exclude Urban_Male and its Male value.
        resolved_svs = [sv.ref for sv in spec.resolved.stat_vars.items]
        assert resolved_svs == [_RURAL_FEMALE_ADULT_SV]
        assert _LEAK_RURAL_MALE_ADULT_SV not in resolved_svs

        gender_result = next(
            s for s in spec.resolved.slots if s.key.property and s.key.property.dcid == "gender"
        )
        assert isinstance(gender_result.binding, BindingValue)
        assert gender_result.binding.value.ref is not None
        assert gender_result.binding.value.ref.dcid == "Female"

        # The minted identity reflects the surviving Female value.
        gender_identity_slot = next(
            s for s in spec.definition.identity.slots if s.property == "gender"
        )
        assert gender_identity_slot.binding.state == "value"
        assert gender_identity_slot.binding.ref == "Female"


# Filter edited set values to the final survivors.

_MV_URBAN_SV = "Count_Person_Urban"
_MV_RURAL_ADULT_SV = "Count_Person_Rural_Adult"
_MV_PERI_ADULT_SV = "Count_Person_Peri_Adult"


def _residence_age_sv_node(label: str, residence: str, age: str | None = None) -> dict:
    constraint_props = [{"dcid": "placeOfResidenceClassification"}]
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "Person"}]},
        "measuredProperty": {"nodes": [{"dcid": "count"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "placeOfResidenceClassification": {"nodes": [{"dcid": residence}]},
    }
    if age is not None:
        constraint_props.append({"dcid": "age"})
        arcs["age"] = {"nodes": [{"dcid": age}]}
    arcs["constraintProperties"] = {"nodes": constraint_props}
    return {"label": label, "type": "StatisticalVariable", "arcs": arcs}


def _mv_gate_graph() -> FakeGraph:
    """Three siblings share a shape. Peri preserves the original Adult union but is narrowed out."""
    nodes = {
        _IND_DCID: {"label": "India", "type": "Country"},
        "Urban": {"label": "Urban", "type": None},
        "Rural": {"label": "Rural", "type": None},
        "Peri": {"label": "Peri-urban", "type": None},
        "Adult": {"label": "Adult", "type": None},
        _MV_URBAN_SV: _residence_age_sv_node("Urban population", "Urban"),
        _MV_RURAL_ADULT_SV: _residence_age_sv_node("Adult rural population", "Rural", age="Adult"),
        _MV_PERI_ADULT_SV: _residence_age_sv_node(
            "Adult peri-urban population", "Peri", age="Adult"
        ),
    }
    obs_entry = [
        {"earliestDate": "2001", "latestDate": "2021", "obsCount": 3, "provenanceId": "dc/base/t"}
    ]
    obs = {
        f"{_MV_URBAN_SV}|{_IND_DCID}": obs_entry,
        f"{_MV_RURAL_ADULT_SV}|{_IND_DCID}": obs_entry,
    }
    return FakeGraph(nodes=nodes, obs=obs)


class TestMultiValueEditFilteredToFinalAsserters:
    def test_residence_edit_survives_only_as_the_final_asserters_value(self):
        """The age gate keeps Rural_Adult, so the edited residence binding excludes Urban."""
        entities = [subject_entity(_IND_DCID)]
        residence_set_slot = _set_slot(
            "how",
            "placeOfResidenceClassification",
            "place of residence",
            ["Urban", "Rural", "Peri"],
        )
        age_slot = _value_slot("how", "age", "age", "Adult")
        base_spec_id, base_identity = make_base(
            PERSON_COUNT_SHAPE, slots=[residence_set_slot, age_slot], entities=entities
        )

        residence_edit_slot = _set_slot(
            "how", "placeOfResidenceClassification", "place of residence", ["Urban", "Rural"]
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                PERSON_COUNT_SHAPE,
                [residence_edit_slot, age_slot],
                entities=entities,
            ),
            base_spec_id=base_spec_id,
            base_identity=base_identity,
            resolved_stat_var_dcids=[_MV_URBAN_SV, _MV_RURAL_ADULT_SV, _MV_PERI_ADULT_SV],
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_mv_gate_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        resolved_svs = [sv.ref for sv in spec.resolved.stat_vars.items]
        assert resolved_svs == [_MV_RURAL_ADULT_SV]

        residence_result = next(
            s
            for s in spec.resolved.slots
            if s.key.property and s.key.property.dcid == "placeOfResidenceClassification"
        )
        assert isinstance(residence_result.binding, BindingValue), (
            f"expected a single-value residence binding of just Rural (the only "
            f"FINAL survivor's own value), got {residence_result.binding!r} -- the "
            "requested Urban must not ride along once its only asserter was "
            "dropped by the untouched-props restriction."
        )
        assert residence_result.binding.value.ref is not None
        assert residence_result.binding.value.ref.dcid == "Rural"

        residence_identity_slot = next(
            s
            for s in spec.definition.identity.slots
            if s.property == "placeOfResidenceClassification"
        )
        assert residence_identity_slot.binding.state == "value"
        assert residence_identity_slot.binding.ref == "Rural"
