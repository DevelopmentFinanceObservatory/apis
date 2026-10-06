"""A resubmit serves the members that assert every bound value.

A ratio-backed identity answers from the member asserting every bound property, and a set
binding elects each member asserting one of its values beside the other bound values.
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
from tests.engine._harness import walk_options
from tests.fixtures import FakeGraph

# Ratio-backed narrow selects a survivor that asserts every retained binding.

_KEN_DCID = "country/KEN"

# Mirrors _mortality_sv_node's own populationType/measuredProperty/statType/
# measurementDenominator arcs below, which the resubmit's walk filters on.
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
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_ratio_gate_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_ratio_gate_graph()),
        )
        assert region.status == "definite", region.no_data_reason
        spec = region.specs[0]

        resolved_svs = [sv.ref.dcid for sv in spec.resolved.members]
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
_RURAL_MALE_ADULT_SV = "Count_Person_Rural_Male_Years18Onwards"


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


def _adult_graph() -> FakeGraph:
    """Urban male, rural female, and adult rural female and male counts in India."""
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
        _RURAL_MALE_ADULT_SV: _person_sv_node(
            "Adult rural male population", "Rural", "Male", age="Years18Onwards"
        ),
    }
    obs_entry = [
        {"earliestDate": "2001", "latestDate": "2021", "obsCount": 3, "provenanceId": "dc/base/t"}
    ]
    obs = {
        f"{_RURAL_FEMALE_ADULT_SV}|{_IND_DCID}": obs_entry,
        f"{_RURAL_MALE_ADULT_SV}|{_IND_DCID}": obs_entry,
    }
    return FakeGraph(nodes=nodes, obs=obs)


class TestASetBindingElectsEveryMemberAssertingTheBoundValues:
    def test_a_gender_set_with_a_bound_age_elects_each_adult_member(self):
        """Rural residence, either gender and adult age: the walk elects the adult rural
        female and male counts, and leaves out the counts that assert no age."""
        entities = [subject_entity(_IND_DCID)]
        residence_slot = _value_slot("how", "placeOfResidenceClassification", "place", "Rural")
        gender_set_slot = _set_slot("how", "gender", "gender", ["Male", "Female"])
        age_value_slot = _value_slot("how", "age", "age", "Years18Onwards")

        inp = ResubmitIdentityPayload(
            identity=make_identity(
                PERSON_COUNT_SHAPE,
                [residence_slot, gender_set_slot, age_value_slot],
                entities=entities,
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_adult_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_adult_graph()),
        )
        assert region.status == "definite", region.no_data_reason
        resolved_svs = {sv.ref.dcid for sv in region.specs[0].resolved.members}
        assert resolved_svs == {_RURAL_FEMALE_ADULT_SV, _RURAL_MALE_ADULT_SV}


# A residence set elects the members asserting the bound age.

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
    """Urban, adult rural and adult peri-urban counts share a shape."""
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


class TestAResidenceSetElectsTheMembersAssertingTheBoundAge:
    def test_a_residence_set_with_a_bound_age_elects_the_adult_rural_count(self):
        """Urban or rural residence, adult age: the walk elects the adult rural count. The
        urban count asserts no age, and the peri-urban one sits outside the set."""
        entities = [subject_entity(_IND_DCID)]
        age_slot = _value_slot("how", "age", "age", "Adult")
        residence_set_slot = _set_slot(
            "how", "placeOfResidenceClassification", "place of residence", ["Urban", "Rural"]
        )
        inp = ResubmitIdentityPayload(
            identity=make_identity(
                PERSON_COUNT_SHAPE, [residence_set_slot, age_slot], entities=entities
            ),
        )
        region = resolve_spec_resubmit(
            inp=inp,
            graph=_mv_gate_graph(),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
            options=walk_options(_mv_gate_graph()),
        )
        assert region.status == "definite", region.no_data_reason
        resolved_svs = [sv.ref.dcid for sv in region.specs[0].resolved.members]
        assert resolved_svs == [_MV_RURAL_ADULT_SV]
