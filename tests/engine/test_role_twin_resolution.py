"""Test search's role rule on IDS-style total/by-creditor twin groups.

A total declares `financingRecipient`; its by-creditor twin declares
`financingProvider` and `financingRecipient` under the same constraints. The
group derives one shape carrying both roles, and admission keeps the member
that answers the roles the query binds.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from search.engine.bind import _UniversalBindOutput
from search.engine.errors import RoleSignatureError
from search.engine.region.candidate_measure_fold import _shapes_are_measure_siblings
from search.engine.region.pipeline import resolve_variable
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.facts import GroupRoles, group_roles
from search.engine.retrieval.retrieve import Materialised, NoDataDraft, materialise
from search.engine.retrieval.role_resolve import graph_confirm_resolve
from search.engine.shape import (
    ShapeDraft,
    declares_shape_roles,
    is_role_bearing,
    required_role_names,
    role_names_by_axis,
)
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import (
    FakeCoverageService,
    build_coverage_index,
    empty_coverage_snapshot,
)
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_TOTAL = "Amount_DebtFixture_PublicOrPubliclyGuaranteed_Nominal"
_BY_CREDITOR = "Amount_DebtFixture_PublicOrPubliclyGuaranteed_ByCreditor_Nominal"
_LENDER_NAMED = "Amount_DebtFixture_IDSNonConcessional_Nominal"

_RECIPIENT_ONLY = RoleNames(donor=None, recipient="financingRecipient")
_BOTH_ROLES = RoleNames(donor="financingProvider", recipient="financingRecipient")
_YEARS = [2019, 2020, 2021]


def _debt_arcs(roles: list[str], *, lender: str = "AllCreditors") -> dict:
    return {
        "populationType": {"nodes": [{"dcid": "DebtFixture"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "constraintProperties": {"nodes": [{"dcid": "idsCreditorType"}]},
        "idsCreditorType": {"nodes": [{"dcid": lender}]},
        "observationProperties": {"nodes": [{"dcid": role} for role in roles]},
    }


def _shape(nodes: dict[str, dict], order: list[str]) -> ShapeDraft:
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    shapes = derive_shapes(confirmed_svs=order, graph=graph)
    assert len(shapes) == 1
    return shapes[0]


def _twin_nodes() -> dict[str, dict]:
    return {
        _TOTAL: {"label": "Debt total", "arcs": _debt_arcs(["financingRecipient"])},
        _BY_CREDITOR: {
            "label": "Debt by creditor",
            "arcs": _debt_arcs(["financingProvider", "financingRecipient"]),
        },
    }


def _twin_coverage() -> FakeCoverageService:
    index = build_coverage_index(
        [
            (_TOTAL, "f_total", "", "country/KEN", _YEARS),
            (_BY_CREDITOR, "f_creditor", "country/CHN", "country/KEN", _YEARS),
        ],
        role_names={_TOTAL: _RECIPIENT_ONLY, _BY_CREDITOR: _BOTH_ROLES},
    )
    return FakeCoverageService(index)


@pytest.mark.parametrize("order", [[_TOTAL, _BY_CREDITOR], [_BY_CREDITOR, _TOTAL]])
def test_twin_group_derives_one_shape_with_the_creditor_role_optional(order) -> None:
    shape = _shape(_twin_nodes(), order)

    assert shape.declared_roles == ("financingProvider", "financingRecipient")
    assert shape.optional_roles == frozenset({"financingProvider"})
    assert required_role_names(shape) == ("financingRecipient",)
    assert role_names_by_axis(shape) == {
        "origin": "financingProvider",
        "destination": "financingRecipient",
    }


def test_twin_shape_accepts_either_member_role_set() -> None:
    shape = _shape(_twin_nodes(), [_TOTAL, _BY_CREDITOR])

    assert declares_shape_roles(shape, ("financingRecipient",))
    assert declares_shape_roles(shape, ("financingRecipient", "financingProvider"))
    assert not declares_shape_roles(shape, ("financingProvider",))
    assert not declares_shape_roles(shape, ("financingRecipient", "creditor"))


def test_optional_roles_keep_measure_siblings_apart() -> None:
    shape = _shape(_twin_nodes(), [_TOTAL, _BY_CREDITOR])
    sibling = dataclasses.replace(shape, meas_prop_dcid="debtService")

    assert _shapes_are_measure_siblings(shape, sibling)
    assert not _shapes_are_measure_siblings(
        shape, dataclasses.replace(sibling, optional_roles=frozenset())
    )


def test_agreeing_group_has_no_optional_role() -> None:
    arcs = _debt_arcs(["financingProvider", "financingRecipient"])
    roles = group_roles({"a": arcs, "b": arcs})

    assert roles == GroupRoles(
        declared=("financingProvider", "financingRecipient"), optional=frozenset()
    )


def test_subject_only_member_beside_a_role_bearing_one_is_refused() -> None:
    subject_only = _debt_arcs([])
    flow = _debt_arcs(["financingProvider", "financingRecipient"])

    with pytest.raises(RoleSignatureError):
        group_roles({"a": subject_only, "b": flow})


# Nominal GDP and a sector's gross value added declare no role; two IDS country series on
# the same EconomicActivity/amount/measuredValue/Nominal five-tuple declare the debtor as
# `financingRecipient`, as on the dev graph.
_GDP_NOMINAL = "Amount_EconomicActivity_GrossDomesticProduction_Nominal"
_GVA_CONSTRUCTION = "Nominal_Amount_EconomicActivity_GrossValueAdded_ConstructionSector"
_GNI = "Amount_EconomicActivity_GrossNationalIncome_Nominal"
_EXPORTS = "Amount_EconomicActivity_ExportsOfGoodsServicesAndPrimaryIncome_Nominal"


def _economic_activity_node(
    name: str, constraints: dict[str, str], roles: tuple[str, ...] = ()
) -> dict:
    arcs = {
        "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
        "name": {"nodes": [{"value": name}]},
        "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }
    if roles:
        arcs["observationProperties"] = {"nodes": [{"dcid": role} for role in roles]}
    return {"label": name, "type": "StatisticalVariable", "arcs": arcs}


def _gdp_nodes() -> dict[str, dict]:
    return {
        _GDP_NOMINAL: _economic_activity_node(
            "Nominal gross domestic product", {"activitySource": "GrossDomesticProduction"}
        ),
        _GVA_CONSTRUCTION: _economic_activity_node(
            "Gross value added in construction sector at current prices",
            {"activitySource": "GrossValueAdded", "economicSector": "ConstructionSector"},
        ),
        _GNI: _economic_activity_node(
            "Gross national income (current US$)",
            {
                "economicActivityType": "GrossNationalIncome",
                "sourceIndicatorCode": "NY.GNP.MKTP.CD",
            },
            ("financingRecipient",),
        ),
        _EXPORTS: _economic_activity_node(
            "Exports of goods, services and primary income (current US$)",
            {
                "economicActivityType": "ExportsOfGoodsServicesAndPrimaryIncome",
                "sourceIndicatorCode": "BX.GSR.TOTL.CD",
            },
            ("financingRecipient",),
        ),
    }


@pytest.mark.parametrize("reverse", [False, True], ids=["recall-order", "reversed"])
def test_subject_only_members_split_from_role_bearing_members_of_one_five_tuple(
    reverse: bool,
) -> None:
    """Members that declare no role and members that declare `financingRecipient` have no
    single role signature, so derive builds one shape for each instead of skipping both.
    The subject-only shape keeps the five-tuple's `shape_id` and the role-bearing one takes
    a suffix."""
    nodes = _gdp_nodes()
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    confirmed = list(nodes)[::-1] if reverse else list(nodes)

    shapes = derive_shapes(confirmed_svs=confirmed, graph=graph)

    assert len(shapes) == 2
    subject, flow = sorted(shapes, key=lambda shape: is_role_bearing(shape))
    assert set(subject.sv_arc_facts or {}) == {_GDP_NOMINAL, _GVA_CONSTRUCTION}
    assert subject.declared_roles == ("observationAbout",)
    assert not is_role_bearing(subject)
    assert subject.optional_roles == frozenset()
    assert set(flow.sv_arc_facts or {}) == {_GNI, _EXPORTS}
    assert flow.declared_roles == ("financingRecipient",)
    assert flow.optional_roles == frozenset()
    assert flow.shape_id == f"{subject.shape_id}_by_role"


class _RecordingBindLLM:
    """Records the bind prompt and elects the first shape."""

    def __init__(self) -> None:
        self.bind_prompts: list[str] = []

    async def generate_structured(self, *, prompt, system, schema):
        assert schema.__name__ == "_UniversalBindOutput"
        self.bind_prompts.append(prompt)
        return _UniversalBindOutput(chosen_shape_index=0, bindings=[]), None


def test_gdp_of_ghana_puts_nominal_gdp_on_the_bind_menu() -> None:
    """The recall pool for "GDP of Ghana", as traced on the dev graph: Nominal GDP ranks
    behind the growth-rate series and ahead of both IDS country series. With the
    five-tuple split on subject-only members, the Nominal GDP shape reaches the menu."""
    nodes = {
        **_gdp_nodes(),
        "GrowthRate_Amount_EconomicActivity_GrossDomesticProduction": {
            "label": "Gross domestic product growth rate",
            "type": "StatisticalVariable",
            "arcs": {
                "typeOf": {"nodes": [{"dcid": "StatisticalVariable"}]},
                "name": {"nodes": [{"value": "Gross domestic product growth rate"}]},
                "populationType": {"nodes": [{"dcid": "EconomicActivity"}]},
                "measuredProperty": {"nodes": [{"dcid": "amount"}]},
                "statType": {"nodes": [{"dcid": "growthRate"}]},
                "constraintProperties": {"nodes": []},
            },
        },
        "country/GHA": {"label": "Ghana", "type": "Country"},
    }
    recalled = [
        "GrowthRate_Amount_EconomicActivity_GrossDomesticProduction",
        _GDP_NOMINAL,
        _GNI,
        _GVA_CONSTRUCTION,
        _EXPORTS,
    ]
    facet = {"earliestDate": "1960", "latestDate": "2024", "obsCount": 65}
    graph = FakeGraph(
        nodes=nodes,
        obs={f"{sv}|country/GHA": [facet] for sv in recalled},
        detect={"GDP": {"svs": recalled, "cosine_scores": [0.7898, 0.7883, 0.7508, 0.72, 0.6967]}},
        resolve={"Ghana": "country/GHA"},
    )
    llm = _RecordingBindLLM()

    asyncio.run(
        resolve_variable(
            "GDP",
            entities=["Ghana"],
            date_request=None,
            detect_query="GDP",
            role_query="GDP",
            pac=True,
            graph=graph,
            resolver=make_resolver(graph),
            llm=llm,
            sdmx=FakeSdmxClient(observations={}),
            base_steps=[],
            base_timing={},
            coverage=empty_coverage_snapshot(),
        )
    )

    assert llm.bind_prompts, "expected the binder to run"
    assert "label: Nominal gross domestic product" in llm.bind_prompts[0]


def test_two_roles_on_one_axis_skip_the_group() -> None:
    """`creditor` and `financingProvider` both name the origin axis."""
    nodes = {
        "Amount_DebtFixture_CreditorA": {
            "label": "A",
            "arcs": _debt_arcs(["creditor", "financingRecipient"]),
        },
        "Amount_DebtFixture_ProviderB": {
            "label": "B",
            "arcs": _debt_arcs(["financingProvider", "financingRecipient"]),
        },
    }
    graph = FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})

    with pytest.raises(RoleSignatureError):
        derive_shapes(confirmed_svs=list(nodes), graph=graph)


def test_debtor_without_creditor_resolves_to_the_total() -> None:
    """Kenya as debtor, no creditor named: the total answers and no donor axis opens."""
    shape = _shape(_twin_nodes(), [_BY_CREDITOR, _TOTAL])

    result = materialise(
        shape,
        [],
        {"financingRecipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_twin_coverage().snapshot(),
        discover_scope=True,
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_TOTAL]
    assert result.open_origin_dcids is None
    assert result.facets_by_sv is not None
    assert [f.facet_id for f in result.facets_by_sv[_TOTAL]] == ["f_total"]


def test_bound_creditor_resolves_to_the_breakdown() -> None:
    shape = _shape(_twin_nodes(), [_TOTAL, _BY_CREDITOR])

    result = materialise(
        shape,
        [],
        {"financingProvider": ["country/CHN"], "financingRecipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_twin_coverage().snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_BY_CREDITOR]
    assert result.facets_by_sv is not None
    assert [f.facet_id for f in result.facets_by_sv[_BY_CREDITOR]] == ["f_creditor"]


def test_admission_requires_a_bound_origin_the_shape_declares_no_role_for() -> None:
    """A bound origin fails step 1 of the role rule on a recipient-only population."""
    shape = _shape({_TOTAL: _twin_nodes()[_TOTAL]}, [_TOTAL])

    result = graph_confirm_resolve(
        shape=shape,
        bindings=[],
        subject_dcids=["country/KEN"],
        origin_dcids=["country/CHN"],
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage_snapshot=_twin_coverage().snapshot(),
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_incompatible"


def test_breakdown_with_an_extra_unbound_constraint_gives_way_to_the_total() -> None:
    """The mint's antichain drops a two-role breakdown whose constraints refine the
    one-role total's, so search resolves the total as the mint does."""
    breakdown_arcs = _debt_arcs(["financingProvider", "financingRecipient"])
    breakdown_arcs["constraintProperties"]["nodes"].append({"dcid": "guaranteeStatus"})
    breakdown_arcs["guaranteeStatus"] = {"nodes": [{"dcid": "PublicOrPubliclyGuaranteed"}]}
    nodes = {
        _TOTAL: {"label": "Debt total", "arcs": _debt_arcs(["financingRecipient"])},
        _BY_CREDITOR: {"label": "Debt by creditor, PPG", "arcs": breakdown_arcs},
    }
    shape = _shape(nodes, [_TOTAL, _BY_CREDITOR])

    result = materialise(
        shape,
        [],
        {"financingRecipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_twin_coverage().snapshot(),
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_TOTAL]


def test_members_with_different_constraints_and_roles_still_refuse() -> None:
    """A two-role lender-named member is no twin of the total, so the survivors disagree."""
    nodes = {
        _TOTAL: {"label": "Debt total", "arcs": _debt_arcs(["financingRecipient"])},
        _LENDER_NAMED: {
            "label": "Debt, non-concessional",
            "arcs": _debt_arcs(
                ["financingProvider", "financingRecipient"], lender="PrivateNonGuaranteedCreditors"
            ),
        },
    }
    shape = _shape(nodes, [_TOTAL, _LENDER_NAMED])

    result = materialise(
        shape,
        [],
        {"financingRecipient": ["country/KEN"]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_twin_coverage().snapshot(),
    )

    assert isinstance(result, NoDataDraft)
    assert result.reason == "role_signature_incompatible"
