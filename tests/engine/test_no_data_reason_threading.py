"""Tests for no-data reason threading through candidate fan-out. Reasons before
`no_observations` in `NO_DATA_RESOLUTION_ORDER` can carry through; later reasons
cannot. A grounded alternate also takes precedence over `variable_not_resolved`.
Place-less tests pin the separate `no_matching_slice` deferral, and a per-capita
refusal before bind that flags the reading the bind model names as nearest.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from search.engine.bind import SlotBindingDraft
from search.engine.clients.llm import _parse
from search.engine.quantity_read import QuantityOutput
from search.engine.region import resolve_variable
from search.engine.region.codes import NO_MEMBER_ASSERTS_BOUND_PROPERTIES
from search.engine.retrieval.retrieve import NoDataDraft, UnsupportedDraft
from search.engine.retrieval.retrieve import materialise as _real_materialise
from tests.engine._binders import elected_reply
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.engine._quantity import NO_QUANTITY
from tests.fixtures import FakeGraph, make_resolver

_ELECTED_SV = "Count_Person_ExperimentalCensus"
_ELECTED_MEAS_PROP = "countExperimentalCensus"
_STD_SV = "HealthcareExpenditureAsFractionOfGDP"

_KEN_NODE = {"country/KEN": {"label": "Kenya", "type": "Country"}}
_ELECTED_NODE = {
    _ELECTED_SV: {
        "label": "Experimental census population count",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Experimental census population count"}]},
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": _ELECTED_MEAS_PROP}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        },
    },
}
_STD_NODE = {
    _STD_SV: {
        "label": "Healthcare expenditure as fraction of GDP",
        "type": "StatisticalVariable",
        "arcs": {
            "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
            "name": {"nodes": [{"value": "Healthcare expenditure as fraction of GDP"}]},
            "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
            "measuredProperty": {"nodes": [{"dcid": "amount"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "measurementDenominator": {"nodes": [{"dcid": "GrossDomesticProduction"}]},
        },
    },
}
_NODES: dict = {**_KEN_NODE, **_ELECTED_NODE, **_STD_NODE}
_DETECT = {
    "experimental census or healthcare expenditure in Kenya": {
        "svs": [_STD_SV, _ELECTED_SV],
        "cosine_scores": [0.85, 0.82],
    },
}


class _ElectedBindLLM:
    """Elects the empty shape and flags the standard alternate."""

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return elected_reply(1, bindings=[], also_plausible=[0]), None


def _run(
    *,
    obs: dict,
    monkeypatch=None,
    elected_reason: str | None = None,
    alternate_unsupported: UnsupportedDraft | None = None,
    alternate_reason: str | None = None,
):
    # Stub the elected shape by its unique measuredProperty. Patch both call
    # sites because pipeline and candidates import `materialise` separately.
    if (
        elected_reason is not None
        or alternate_unsupported is not None
        or alternate_reason is not None
    ):

        def _fake_materialise(shape, *args, **kwargs):
            if shape.meas_prop_dcid == _ELECTED_MEAS_PROP:
                if elected_reason is not None:
                    return NoDataDraft(reason=elected_reason)
            elif alternate_unsupported is not None:
                return alternate_unsupported
            elif alternate_reason is not None:
                return NoDataDraft(reason=alternate_reason)
            return _real_materialise(shape, *args, **kwargs)

        assert monkeypatch is not None
        monkeypatch.setattr("search.engine.region.materialise_step.materialise", _fake_materialise)
        monkeypatch.setattr("search.engine.region.candidates.materialise", _fake_materialise)

    graph = FakeGraph(
        nodes=_NODES,
        detect=_DETECT,
        resolve={"Kenya": "country/KEN"},
        obs=obs,
    )
    return asyncio.run(
        resolve_variable(
            "experimental census or healthcare expenditure in Kenya",
            entities=["Kenya"],
            date_request=None,
            detect_query="experimental census or healthcare expenditure in Kenya",
            role_query="experimental census or healthcare expenditure in Kenya",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_ElectedBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(scope_availability=graph.scope_availability),
            coverage=empty_coverage_snapshot(),
        )
    )


# A capped alternate is unverified and blocks a definite answer; a completed
# no-data alternate allows one. The fake stands in for role-scope's cap path.


# The alternate holds data in Kenya, so the status rule offers it and its read is made.
_BOTH_IN_KENYA = {
    f"{_ELECTED_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
}


def test_capped_alternate_returns_unsupported_not_elected_definite(monkeypatch):
    result = _run(
        obs=_BOTH_IN_KENYA,
        monkeypatch=monkeypatch,
        alternate_unsupported=UnsupportedDraft(
            code="scope_too_broad",
            pointer="/definition/identity/entities/debtor",
            detail="the alternate scope exceeds the deployment member cap",
        ),
    )

    assert result.status == "unsupported", result
    assert result.unsupported is not None
    assert result.unsupported.code == "scope_too_broad"
    assert result.unsupported.pointer == "/definition/identity/entities/debtor"


def test_no_data_alternate_keeps_elected_definite():
    result = _run(
        obs={
            f"{_ELECTED_SV}|country/KEN": [
                {"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}
            ]
        }
    )

    assert result.status == "definite", result


# ---------------------------------------------------------------------------
# With no surviving candidates, the elected shape's failure reason is retained.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("elected_reason", "expected_reason"),
    [
        pytest.param("no_matching_slice", "no_matching_slice", id="earlier-elected-reason"),
        pytest.param("facet_not_available", "no_observations", id="later-elected-reason"),
    ],
)
def test_empty_candidates_uses_reason_order(monkeypatch, elected_reason, expected_reason):
    """Empty fan-out retains only an elected reason ordered before no_observations. The
    alternate holds data in Kenya, so it is read, and its read finds none."""
    result = _run(
        obs={f"{_STD_SV}|country/KEN": _BOTH_IN_KENYA[f"{_STD_SV}|country/KEN"]},
        monkeypatch=monkeypatch,
        elected_reason=elected_reason,
        alternate_reason="no_observations",
    )
    assert result.status == "no_data"
    assert result.no_data_reason == expected_reason


def test_empty_candidates_has_no_flagged_match(monkeypatch):
    result = _run(obs={}, monkeypatch=monkeypatch, elected_reason="no_matching_slice")
    assert result.specs == ()
    assert not result.flagged_match
    assert result.alternate_match is None


# ---------------------------------------------------------------------------
# One alternate survives, below CandidateSet's two-spec minimum, so the result
# returns no data with the elected shape's failure reason.
# ---------------------------------------------------------------------------


def test_sole_survivor_reports_elected_reason_not_no_observations(monkeypatch):
    obs = {
        f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    }
    result = _run(obs=obs, monkeypatch=monkeypatch, elected_reason="no_matching_slice")
    assert result.status == "no_data"
    assert result.no_data_reason == "no_matching_slice", (
        f"expected the elected shape's own no_matching_slice failure, got {result.no_data_reason!r}"
    )
    assert result.alternate_match is not None


# A sole survivor carries an elected reason only when it sorts before
# `no_observations`. No live materialise path emits the later `facet_not_available`.


def test_sole_survivor_facet_not_available_does_not_thread(monkeypatch):
    obs = {
        f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    }
    result = _run(obs=obs, monkeypatch=monkeypatch, elected_reason="facet_not_available")
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations", (
        f"facet_not_available sorts after no_observations and must not thread, "
        f"got {result.no_data_reason!r}"
    )
    assert result.alternate_match is not None


# A grounded alternate proves a StatVar matched the query, so the elected
# `variable_not_resolved` reason gives way to `no_observations`.


def test_sole_survivor_variable_not_resolved_does_not_thread_with_grounded_near_match(
    monkeypatch,
):
    """A grounded near-match sets `no_observations` and preserves the
    alternate match when the elected reason is `variable_not_resolved`.
    """
    obs = {
        f"{_STD_SV}|country/KEN": [{"earliestDate": "2015", "latestDate": "2022", "obsCount": 5}],
    }
    result = _run(obs=obs, monkeypatch=monkeypatch, elected_reason="variable_not_resolved")
    assert result.status == "no_data"
    assert result.no_data_reason == "no_observations", (
        "variable_not_resolved must not thread alongside a grounded near-match, "
        f"got {result.no_data_reason!r}"
    )
    assert result.alternate_match is not None


def test_empty_candidates_variable_not_resolved_threads_without_flagged_match(monkeypatch):
    """With no alternate to attach, the result uses the elected
    `variable_not_resolved` reason.
    """
    result = _run(obs={}, monkeypatch=monkeypatch, elected_reason="variable_not_resolved")
    assert result.status == "no_data"
    assert result.no_data_reason == "variable_not_resolved"
    assert not result.flagged_match
    assert result.alternate_match is None


# ---------------------------------------------------------------------------
# A place-less query with split binding ownership returns no_matching_slice, and a
# place-less per-capita query that no reading serves refuses on the form before bind.
# ---------------------------------------------------------------------------

_PC_PROP_GENDER = "gender"
_PC_VAL_FEMALE = "Female"
_PC_PROP_RESIDENCE = "placeOfResidenceClassification"
_PC_VAL_URBAN = "Urban"

_PC_SV_A = "TEST/PerCapitaSplitOwnership_GenderOnly"
_PC_SV_B = "TEST/PerCapitaSplitOwnership_ResidenceOnly"

_PC_POP_TYPE = "TestPerCapitaIncidenceEvent"
_PC_MEAS_PROP = "testPerCapitaIncidence"
_PC_STAT_TYPE = "measuredValue"

_PC_NO_ENTITY_QUERY = "female urban test measure"
_PC_FULLY_ASSERTED_QUERY = "female test incidence per capita"


def _pc_sv_arcs(*, label: str, prop: str, value: str) -> dict:
    return {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": label}]},
        "populationType": {"nodes": [{"dcid": _PC_POP_TYPE}]},
        "measuredProperty": {"nodes": [{"dcid": _PC_MEAS_PROP}]},
        "statType": {"nodes": [{"dcid": _PC_STAT_TYPE}]},
        "constraintProperties": {"nodes": [{"dcid": prop}]},
        prop: {"nodes": [{"dcid": value}]},
    }


_PC_NODES = {
    _PC_SV_A: {
        "label": "Test per-capita incidence, female (any residence)",
        "type": "StatisticalVariable",
        "arcs": _pc_sv_arcs(
            label="Test per-capita incidence, female (any residence)",
            prop=_PC_PROP_GENDER,
            value=_PC_VAL_FEMALE,
        ),
    },
    _PC_SV_B: {
        "label": "Test per-capita incidence, urban (any gender)",
        "type": "StatisticalVariable",
        "arcs": _pc_sv_arcs(
            label="Test per-capita incidence, urban (any gender)",
            prop=_PC_PROP_RESIDENCE,
            value=_PC_VAL_URBAN,
        ),
    },
    _PC_VAL_FEMALE: {"label": "Female", "type": "GenderType"},
    _PC_VAL_URBAN: {"label": "Urban", "type": "HumanSettlementClassificationEnum"},
}

_PC_NO_ENTITY_DETECT = {
    _PC_NO_ENTITY_QUERY: {
        "svs": [_PC_SV_A, _PC_SV_B],
        "cosine_scores": [0.9, 0.85],
    },
    _PC_FULLY_ASSERTED_QUERY: {
        "svs": [_PC_SV_A],
        "cosine_scores": [0.9],
    },
}

_PC_USAGE = {"input_tokens": 11, "output_tokens": 7, "cached_tokens": 0}


class _PerCapitaSplitOwnershipBindLLM:
    """Binds Female and Urban, which no member asserts together. Returns usage
    so the test also checks `llm_usage` survives the deferral.
    """

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return NO_QUANTITY
        assert schema.__name__ == "_UniversalBindOutput"
        return (
            elected_reply(
                0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid=_PC_PROP_GENDER,
                        kind="value",
                        value_dcids=[_PC_VAL_FEMALE],
                    ),
                    SlotBindingDraft(
                        axis="how",
                        property_dcid=_PC_PROP_RESIDENCE,
                        kind="value",
                        value_dcids=[_PC_VAL_URBAN],
                    ),
                ],
                also_plausible=[],
            ),
            dict(_PC_USAGE),
        )


_DECLINED_REPLY = {
    "chosen_shape_index": 0,
    "ask": "no shape on the menu measures this per person",
    "nearest_refused_index": 0,
}


class _PerCapitaDeclinedBindLLM:
    """Answers the quantity read with the per-person form, which leaves the sole reading not
    electable, and declines in bind with `reply`, naming that reading as nearest. The reply
    goes through the parse a live call runs (`clients.llm._parse`)."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        if schema is QuantityOutput:
            return QuantityOutput(stated_form="per_person"), None
        assert schema.__name__ == "_UniversalBindOutput"
        self.prompts.append(prompt)
        return _parse(json.dumps(self.reply), schema, model="fake", usage=None), None


def test_place_less_split_ownership_no_matching_slice_defers_to_mint_with_flagged_match():
    """With no member asserting both bound values, the place-less mint refuses
    `no_matching_slice` and preserves the flagged match and LLM usage.
    """
    graph = FakeGraph(nodes=_PC_NODES, detect=_PC_NO_ENTITY_DETECT, resolve={}, obs={})
    result = asyncio.run(
        resolve_variable(
            _PC_NO_ENTITY_QUERY,
            entities=[],
            date_request=None,
            detect_query=_PC_NO_ENTITY_QUERY,
            role_query=_PC_NO_ENTITY_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=_PerCapitaSplitOwnershipBindLLM(),
            base_timing={},
            sdmx=FakeSdmxClient(
                availability={"recipient": ["country/KEN"]},
                scope_availability=graph.scope_availability,
            ),
            coverage=empty_coverage_snapshot(),
        )
    )

    assert result.status == "no_data"
    assert result.no_data_reason == "no_matching_slice", (
        f"expected the deferral to elect the pipeline's own place-less mint's "
        f"no_matching_slice, got {result.no_data_reason!r}"
    )
    assert result.flagged_match is not None and len(result.flagged_match) == 1
    assert result.gate == NO_MEMBER_ASSERTS_BOUND_PROPERTIES
    assert result.llm_usage == _PC_USAGE


@pytest.mark.parametrize(
    "reply",
    [
        _DECLINED_REPLY,
        {**_DECLINED_REPLY, "measuring": [{"index": 0, "rating": 1.0}]},
    ],
    ids=["asks", "asks-and-rates-the-nearest-refused"],
)
def test_place_less_per_capita_fully_asserted_member_reports_denominator_not_available(reply):
    """The member asserts Female and has a 2020 Kenya observation, but it is no per-capita
    series, so the per-person form the quantity read states leaves it not electable. Bind
    declines and names it as nearest, so the query reports `denominator_not_available`,
    flagging that member, before any read. The menu is empty, so a reply that also rates
    the nearest refused reading answers the same.
    """
    graph = FakeGraph(
        nodes=_PC_NODES,
        detect=_PC_NO_ENTITY_DETECT,
        resolve={},
        obs={
            f"{_PC_SV_A}|country/KEN": [
                {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
            ]
        },
    )
    llm = _PerCapitaDeclinedBindLLM(reply)
    result = asyncio.run(
        resolve_variable(
            _PC_FULLY_ASSERTED_QUERY,
            entities=[],
            date_request=None,
            detect_query=_PC_FULLY_ASSERTED_QUERY,
            role_query=_PC_FULLY_ASSERTED_QUERY,
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm,
            base_timing={},
            sdmx=FakeSdmxClient(
                availability={"recipient": ["country/KEN"]},
                scope_availability=graph.scope_availability,
            ),
            coverage=empty_coverage_snapshot(),
        )
    )

    (prompt,) = llm.prompts
    assert "(fails the form the query asks for)" in prompt.partition("NOT ELECTABLE")[2]
    assert result.status == "no_data"
    assert result.no_data_reason == "denominator_not_available"
    (flagged,) = result.flagged_match or ()
    assert [sv.ref.ref for sv in flagged.resolved.members] == [_PC_SV_A]
    assert "materialise" not in result.timing_by_step
