"""Tests the label `derive_shapes` gives a shape: a multi-member UN System family
reads as its population type plus the values every member shares, and every
other shape keeps its member's name."""

from __future__ import annotations

from search.engine.retrieval.derive import derive_shapes
from tests.fixtures import FakeGraph

_CME = "undata/unicef/SERIES-CME_MR"
_NUT = "undata/who/SERIES-NUT_ANTH_SH"

_UN_LABELS = {
    _CME: "Child mortality rate",
    _NUT: "Prevalence of malnutrition among children",
    "undataAge": "Age",
    "undataSex": "Sex",
    "undataNutStatus": "Nutritional status",
    "undata/AGE-Y0T4": "Under 5 years old",
    "undata/AGE-Y0": "Under 1 year old",
    "undata/AGE-M24T59": "24 to 59 months old",
    "undata/AGE-M0T23": "0 to 23 months old",
    "undata/SEX-M": "Male",
    "undata/SEX-F": "Female",
    "undata/NUT_STATUS-STUNTING": "Stunting (height-for-age <-2 SD)",
}


def _un_member(name: str, population_type: str, constraints: dict[str, str]) -> dict:
    arcs: dict = {
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": population_type}]},
        "measuredProperty": {"nodes": [{"dcid": "value"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
    }
    for prop, value in constraints.items():
        arcs[prop] = {"nodes": [{"dcid": value}]}
    return {"label": name, "arcs": arcs}


def _graph(members: dict[str, dict]) -> FakeGraph:
    nodes = {dcid: {"label": label} for dcid, label in _UN_LABELS.items()}
    nodes.update(members)
    return FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})


def _only_shape(members: dict[str, dict], recalled: list[str]) -> str:
    shapes = derive_shapes(
        confirmed_svs=list(members),
        graph=_graph(members),
        sv_scores={sv: 0.9 - i * 0.01 for i, sv in enumerate(recalled)},
    )
    assert len(shapes) == 1
    return shapes[0].label


def test_un_family_spanning_sex_and_age_reads_as_its_population_type():
    """The recalled member's Age and Sex values stay out of the family label."""
    members = {
        "undata/unicef/CME_MR.AGE--Y0T4__SEX--M": _un_member(
            "Child mortality rate [Age=Under 5 years old, Sex=Male]",
            _CME,
            {"undataAge": "undata/AGE-Y0T4", "undataSex": "undata/SEX-M"},
        ),
        "undata/unicef/CME_MR.AGE--Y0T4__SEX--F": _un_member(
            "Child mortality rate [Age=Under 5 years old, Sex=Female]",
            _CME,
            {"undataAge": "undata/AGE-Y0T4", "undataSex": "undata/SEX-F"},
        ),
        "undata/unicef/CME_MR.AGE--Y0__SEX--M": _un_member(
            "Child mortality rate [Age=Under 1 year old, Sex=Male]",
            _CME,
            {"undataAge": "undata/AGE-Y0", "undataSex": "undata/SEX-M"},
        ),
    }

    label = _only_shape(members, recalled=["undata/unicef/CME_MR.AGE--Y0T4__SEX--M"])

    assert label == "Child mortality rate"


def test_un_family_keeps_the_value_every_member_shares():
    """Nutritional status is Stunting for every member; Age varies and stays out."""
    members = {
        "undata/who/NUT_ANTH_SH.AGE--M24T59__NUT_STATUS--STUNTING": _un_member(
            "Prevalence of malnutrition among children "
            "[Age=24 to 59 months old, Nutritional status=Stunting (height-for-age <-2 SD)]",
            _NUT,
            {"undataAge": "undata/AGE-M24T59", "undataNutStatus": "undata/NUT_STATUS-STUNTING"},
        ),
        "undata/who/NUT_ANTH_SH.AGE--M0T23__NUT_STATUS--STUNTING": _un_member(
            "Prevalence of malnutrition among children "
            "[Age=0 to 23 months old, Nutritional status=Stunting (height-for-age <-2 SD)]",
            _NUT,
            {"undataAge": "undata/AGE-M0T23", "undataNutStatus": "undata/NUT_STATUS-STUNTING"},
        ),
    }

    label = _only_shape(
        members, recalled=["undata/who/NUT_ANTH_SH.AGE--M24T59__NUT_STATUS--STUNTING"]
    )

    assert label == (
        "Prevalence of malnutrition among children "
        "[Nutritional status=Stunting (height-for-age <-2 SD)]"
    )


def test_single_member_un_shape_keeps_its_member_name():
    member = "undata/unicef/CME_MR.AGE--Y0T4__SEX--M"
    members = {
        member: _un_member(
            "Child mortality rate [Age=Under 5 years old, Sex=Male]",
            _CME,
            {"undataAge": "undata/AGE-Y0T4", "undataSex": "undata/SEX-M"},
        )
    }

    label = _only_shape(members, recalled=[member])

    assert label == "Child mortality rate [Age=Under 5 years old, Sex=Male]"


def test_base_family_keeps_its_recalled_member_name():
    """A base Data Commons family keeps its recalled member's name, as it always has,
    even when that name carries a gender the family spans."""

    def base_member(name: str, gender: str) -> dict:
        return {
            "label": name,
            "arcs": {
                "name": {"nodes": [{"value": name}]},
                "populationType": {"nodes": [{"dcid": "Person"}]},
                "measuredProperty": {"nodes": [{"dcid": "lifeExpectancy"}]},
                "statType": {"nodes": [{"dcid": "measuredValue"}]},
                "constraintProperties": {"nodes": [{"dcid": "gender"}]},
                "gender": {"nodes": [{"dcid": gender}]},
            },
        }

    members = {
        "LifeExpectancy_Person_Female": base_member("Female life expectancy", "Female"),
        "LifeExpectancy_Person_Male": base_member("Male life expectancy", "Male"),
    }
    graph = FakeGraph(
        nodes={"Person": {"label": "Person"}, "gender": {"label": "gender"}, **members},
        obs={},
        detect={},
        resolve={},
    )

    shapes = derive_shapes(
        confirmed_svs=list(members),
        graph=graph,
        sv_scores={"LifeExpectancy_Person_Female": 0.9, "LifeExpectancy_Person_Male": 0.8},
    )

    assert [shape.label for shape in shapes] == ["Female life expectancy"]
