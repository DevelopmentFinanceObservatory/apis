"""Tests for standard multi-axis set bindings.

A member must match one set value without contradicting another axis. Even after fail-open,
only members that assert every bound property can win.
"""

from __future__ import annotations

from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.retrieve import BoundValuesUnrealisable, Materialised, materialise
from search.engine.shape import ShapeDraft
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_RECIPIENT = "country/USA"

_SOURCE_PROP = "financingSource"
_TYPE_PROP = "expenditureType"
_SOURCE_VALS = ["SourceA", "SourceB"]
_TYPE_VALS = ["TypeX", "TypeY"]


def _two_axis_arcs(*, source: str | None = None, exp_type: str | None = None) -> dict:
    """Arcs for a member that may carry either or both of the two constraint props."""
    constraint_props = []
    arcs: dict = {
        "populationType": {"nodes": [{"dcid": "TestExpenditureActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "testExpenditure"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
    }
    if source is not None:
        constraint_props.append(_SOURCE_PROP)
        arcs[_SOURCE_PROP] = {"nodes": [{"dcid": source}]}
    if exp_type is not None:
        constraint_props.append(_TYPE_PROP)
        arcs[_TYPE_PROP] = {"nodes": [{"dcid": exp_type}]}
    arcs["constraintProperties"] = {"nodes": [{"dcid": p} for p in constraint_props]}
    return arcs


def _shape_with(sv_arc_facts: dict[str, dict]) -> ShapeDraft:
    return ShapeDraft(
        shape_id="test_multi_set_axes_shape",
        label="test multi-set-axes shape",
        pop_type_dcid="TestExpenditureActivity",
        meas_prop_dcid="testExpenditure",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        slot_keys=(),
        sv_arc_facts=sv_arc_facts,
        representative_score=0.9,
    )


def _graph_with_data_for(*svs: str) -> FakeGraph:
    obs = {
        f"{sv}|{_RECIPIENT}": [{"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}]
        for sv in svs
    }
    return FakeGraph(nodes={}, obs=obs, detect={}, resolve={})


def _two_set_bindings() -> list[SlotBindingDraft]:
    return [
        SlotBindingDraft(
            axis="how",
            property_dcid=_SOURCE_PROP,
            kind="set",
            value_dcids=list(_SOURCE_VALS),
        ),
        SlotBindingDraft(
            axis="how",
            property_dcid=_TYPE_PROP,
            kind="set",
            value_dcids=list(_TYPE_VALS),
        ),
    ]


class TestMultiSetAxesCoherentCase:
    """A winning member must assert both set-bound properties."""

    def test_only_the_member_asserting_both_axes_survives(self):
        sv_both = "SV_both_axes_match"  # SourceA + TypeX: asserts both bound axes
        sv_source_only = "SV_source_only"  # SourceA, no expenditureType: silent on one axis
        sv_bad_source = "SV_bad_source"  # SourceC (not in set) + TypeX

        shape = _shape_with(
            {
                sv_both: _two_axis_arcs(source="SourceA", exp_type="TypeX"),
                sv_source_only: _two_axis_arcs(source="SourceA"),
                sv_bad_source: _two_axis_arcs(source="SourceC", exp_type="TypeX"),
            }
        )
        graph = _graph_with_data_for(sv_both, sv_source_only, sv_bad_source)

        result = materialise(
            shape,
            _two_set_bindings(),
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.has_data is True
        # sv_bad_source explicitly contradicts the source axis (SourceC not in
        # {SourceA, SourceB}) -- it must never be collected as a co-answer.
        # sv_source_only is silent on the expenditureType axis -- asserting
        # only one of two simultaneously-bound axes is not enough to win;
        # sv_both, which asserts both, is the sole survivor.
        assert result.sv_dcids == [sv_both]


class TestMultiSetAxesUnrealisable:
    """Members that contradict different set axes are not combined."""

    def test_discordant_members_are_not_combined_as_a_false_co_answer(self):
        # sv_a matches the source axis but explicitly contradicts the type axis
        # (TypeZ is outside {TypeX, TypeY}). sv_b is the mirror image: matches
        # the type axis but explicitly contradicts the source axis (SourceC is
        # outside {SourceA, SourceB}). Neither member is a coherent multi-axis
        # match, so no member survives the bound-value filter and the resolve
        # refuses, naming both axes for the caller to unbind.
        sv_a = "SV_matches_source_only"
        sv_b = "SV_matches_type_only"

        shape = _shape_with(
            {
                sv_a: _two_axis_arcs(source="SourceA", exp_type="TypeZ"),
                sv_b: _two_axis_arcs(source="SourceC", exp_type="TypeX"),
            }
        )
        graph = _graph_with_data_for(sv_a, sv_b)

        result = materialise(
            shape,
            _two_set_bindings(),
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, BoundValuesUnrealisable)
        assert [c.properties for c in result.conflicts] == [{_TYPE_PROP}, {_SOURCE_PROP}]

    def test_member_matching_neither_axis_explicitly_is_not_collected(self):
        """A sibling with neither property has no explicit set match."""
        sv_source_only = "SV_source_only_2"  # matches source axis explicitly
        sv_unrelated = "SV_unrelated"  # neither property at all

        shape = _shape_with(
            {
                sv_source_only: _two_axis_arcs(source="SourceA"),
                sv_unrelated: _two_axis_arcs(),
            }
        )
        graph = _graph_with_data_for(sv_source_only, sv_unrelated)

        result = materialise(
            shape,
            _two_set_bindings(),
            {"observationAbout": [_RECIPIENT]},
            graph=graph,
            sdmx=FakeSdmxClient(observations={}, availability={}),
            coverage=empty_coverage_snapshot(),
        )

        assert isinstance(result, Materialised)
        assert result.sv_dcids == [sv_source_only]
