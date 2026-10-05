"""Checks fresh-path `inapplicable_slots` against each resolved StatVar's graph arcs.

Three siblings share one shape. The fixture tests absent bindings, a bound property, and a
value-mismatched asserter. No sibling satisfies every bound property, so the result is flagged
`no_data`.
"""

from __future__ import annotations

import asyncio

from search.engine.bind import SlotBindingDraft, _UniversalBindOutput
from search.engine.region import resolve_variable
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_QUERY = "domestic government health spending in Kenya"

_PROP_FIN_SOURCE = "healthFinancingSource"
_VAL_DOMESTIC_GOVT = "DomesticGeneralGovernmentHealthFinancing"
_PROP_FUNCTION = "healthCareFunction"
_VAL_FUNCTION = "HC_6_1"
_PROP_AGE = "ageGroup"
# What SV_BY_AGE's own arcs actually assert:
_VAL_AGE_ASSERTED = "AgeGroup_18OrMore"
# The bound value mismatches SV_BY_AGE, so that sibling cannot satisfy the member probe.
_VAL_AGE_BOUND = "AgeGroup_65OrMore"

_SV_DOMESTIC_GOVT = "TEST/HealthSpend_DomesticGovt"
_SV_BY_FUNCTION = "TEST/HealthSpend_ByFunction"
_SV_BY_AGE = "TEST/HealthSpend_ByAge"

_KEN_DCID = "country/KEN"


def _sv_arcs(*, label: str, constraint_prop: str, constraint_value: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": "HealthExpenditureEvent"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": constraint_prop}]},
        constraint_prop: {"nodes": [{"dcid": constraint_value}]},
    }


_NODES = {
    _KEN_DCID: {"label": "Kenya", "type": "Country"},
    _SV_DOMESTIC_GOVT: {
        "label": "Domestic general government health expenditure",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Domestic general government health expenditure",
            constraint_prop=_PROP_FIN_SOURCE,
            constraint_value=_VAL_DOMESTIC_GOVT,
        ),
    },
    _SV_BY_FUNCTION: {
        "label": "Health expenditure by function",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Health expenditure by function",
            constraint_prop=_PROP_FUNCTION,
            constraint_value=_VAL_FUNCTION,
        ),
    },
    _SV_BY_AGE: {
        "label": "Health expenditure by age group",
        "type": "StatisticalVariable",
        "arcs": _sv_arcs(
            label="Health expenditure by age group",
            constraint_prop=_PROP_AGE,
            constraint_value=_VAL_AGE_ASSERTED,
        ),
    },
    # Needed to ground the (oddly-drafted) ageGroup="value" binding below --
    # build_slot falls back to unbound when a value dcid doesn't resolve.
    _VAL_AGE_BOUND: {"label": "65 years or more", "type": "Thing"},
}

_DETECT = {
    _QUERY: {
        "svs": [_SV_DOMESTIC_GOVT, _SV_BY_FUNCTION, _SV_BY_AGE],
        "cosine_scores": [0.9, 0.85, 0.8],
    },
}

# Observations keep all three properties in the shape.
# The mismatched age binding leaves no sibling for the member probe.
_OBS = {
    f"{_SV_DOMESTIC_GOVT}|{_KEN_DCID}": [
        {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
    ],
    f"{_SV_BY_FUNCTION}|{_KEN_DCID}": [
        {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
    ],
    f"{_SV_BY_AGE}|{_KEN_DCID}": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


class _AbsentMisclassifyLLM:
    """Drafts two absent bindings and a mismatched ageGroup value."""

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        return _UniversalBindOutput(
            chosen_shape_index=0,
            bindings=[
                SlotBindingDraft(
                    axis="how", property_dcid=_PROP_FIN_SOURCE, kind="absent", value_dcids=[]
                ),
                SlotBindingDraft(
                    axis="how", property_dcid=_PROP_FUNCTION, kind="absent", value_dcids=[]
                ),
                SlotBindingDraft(
                    axis="how",
                    property_dcid=_PROP_AGE,
                    kind="value",
                    value_dcids=[_VAL_AGE_BOUND],
                ),
            ],
            also_plausible=[],
            ask=None,
        ), None


def _run():
    graph = FakeGraph(nodes=_NODES, detect=_DETECT, resolve={"Kenya": _KEN_DCID}, obs=_OBS)
    return asyncio.run(
        resolve_variable(
            _QUERY,
            entities=["Kenya"],
            date_request=None,
            detect_query=_QUERY,
            role_query=_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            sdmx=FakeSdmxClient(observations={}, availability={}),
            llm=_AbsentMisclassifyLLM(),
            base_steps=[],
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )


def test_selected_statvar_asserts_the_property_so_it_is_not_inapplicable():
    """The resolved StatVar's arcs keep its declared financing source out of `inapplicable_slots`."""
    result = _run()
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None
    spec = result.flagged_match[0]
    assert [sv.ref for sv in spec.resolved.stat_vars.items] == [_SV_DOMESTIC_GOVT]
    assert _PROP_FIN_SOURCE not in spec.resolved.inapplicable_slots
    # Still unbound on the wire (the query never named a financing source) --
    # this is "unbound, and not inapplicable", not a bound slot.
    fin_source_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.dcid == _PROP_FIN_SOURCE
    )
    assert fin_source_slot.binding.kind == "unbound"


def test_genuinely_undeclared_property_is_still_reported():
    """The true-positive case: healthCareFunction was
    ALSO drafted "absent", and the resolved StatVar (SV_DOMESTIC_GOVT) really
    doesn't assert it -- only the never-selected sibling SV_BY_FUNCTION does.
    "no resolved StatVar declares this property" is true here, so it must
    still appear."""
    result = _run()
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None
    spec = result.flagged_match[0]
    assert _PROP_FUNCTION in spec.resolved.inapplicable_slots


def test_bound_slot_is_never_reported_inapplicable_even_if_drafted_oddly():
    """The bound ageGroup slot stays out of `inapplicable_slots` even without a matching arc."""
    result = _run()
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None
    spec = result.flagged_match[0]
    age_slot = next(
        s
        for s in spec.resolved.slots
        if s.key.property is not None and s.key.property.dcid == _PROP_AGE
    )
    assert age_slot.binding.kind == "value"
    assert _PROP_AGE not in spec.resolved.inapplicable_slots


def test_no_property_is_both_bound_and_inapplicable():
    """The same coherence invariant test_resubmit_inapplicable_slots.py
    checks after every resubmit scenario, checked here for the fresh path."""
    result = _run()
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations"
    assert result.flagged_match is not None
    spec = result.flagged_match[0]
    bound_props = {
        s.key.property.dcid
        for s in spec.resolved.slots
        if s.key.property is not None and s.binding.kind != "unbound"
    }
    assert not (set(spec.resolved.inapplicable_slots) & bound_props)
