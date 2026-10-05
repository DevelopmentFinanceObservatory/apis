"""A role-bearing answer serves only members that assert every bound property.

The IDS debt-service family on the dev graph: the public or publicly guaranteed (PPG)
total asserts `guaranteeStatus`, while the creditor-type breakdowns of public-sector debt
leave it off. A query binding `guaranteeStatus=PublicOrPubliclyGuaranteed` must serve the
PPG total alone, as the code index, the mint and the resubmit already require.
"""

from __future__ import annotations

from search.engine.bind import SlotBindingDraft
from search.engine.retrieval.derive import derive_shapes
from search.engine.retrieval.retrieve import Materialised, NoDataDraft, materialise
from search.engine.shape import ShapeDraft
from series_coverage.schema import RoleNames
from tests.engine._fake_coverage import FakeCoverageService, build_coverage_index
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph

_KEN = "country/KEN"
_PPG = "PublicOrPubliclyGuaranteed"
_PPG_TOTAL = "Amount_FinancialFlow_ServicingPayment_PublicOrPubliclyGuaranteed_Nominal"
_PPG_PUBLIC_SECTOR = (
    "Amount_FinancialFlow_ServicingPayment_PublicOrPubliclyGuaranteed_IDSDebtorTypePS_Nominal"
)
_BONDS_PUBLIC_SECTOR = (
    "Amount_FinancialFlow_ServicingPayment_IDSCreditorTypePBND_IDSDebtorTypePS_Nominal"
)
_PRIVATE_GENERAL_GOVERNMENT = (
    "Amount_FinancialFlow_ServicingPayment_IDSCreditorTypePRVT_IDSDebtorTypeGG_Nominal"
)
_YEARS = [2022, 2023, 2024]


def _arcs(indicator: str, constraints: dict[str, str]) -> dict:
    """A member's arcs as the dev graph serves them: recipient role only."""
    constraints = {"financialFlowType": "ServicingPayment", **constraints}
    constraints["sourceIndicatorCode"] = indicator
    return {
        "populationType": {"nodes": [{"dcid": "FinancialFlow"}]},
        "measuredProperty": {"nodes": [{"dcid": "amount"}]},
        "statType": {"nodes": [{"dcid": "measuredValue"}]},
        "measurementQualifier": {"nodes": [{"dcid": "Nominal"}]},
        "constraintProperties": {"nodes": [{"dcid": prop} for prop in constraints]},
        "observationProperties": {"nodes": [{"dcid": "financingRecipient"}]},
        **{prop: {"nodes": [{"dcid": value}]} for prop, value in constraints.items()},
    }


_MEMBERS = {
    _PPG_TOTAL: _arcs("DT.TDS.DPPG.CD", {"guaranteeStatus": _PPG}),
    _PPG_PUBLIC_SECTOR: _arcs(
        "DT.TDS.DEPS.CD", {"guaranteeStatus": _PPG, "idsDebtorType": "IDSDebtorTypePS"}
    ),
    _BONDS_PUBLIC_SECTOR: _arcs(
        "DT.TDS.PBND.PS.CD",
        {"idsCreditorType": "IDSCreditorTypePBND", "idsDebtorType": "IDSDebtorTypePS"},
    ),
    _PRIVATE_GENERAL_GOVERNMENT: _arcs(
        "DT.TDS.PRVT.GG.CD",
        {"idsCreditorType": "IDSCreditorTypePRVT", "idsDebtorType": "IDSDebtorTypeGG"},
    ),
}


def _shape(members: dict[str, dict]) -> ShapeDraft:
    nodes = {sv: {"label": sv, "arcs": arcs} for sv, arcs in members.items()}
    shapes = derive_shapes(
        confirmed_svs=list(members), graph=FakeGraph(nodes=nodes, obs={}, detect={}, resolve={})
    )
    assert len(shapes) == 1, [s.shape_id for s in shapes]
    return shapes[0]


def _coverage(svs: list[str]) -> FakeCoverageService:
    index = build_coverage_index(
        [(sv, f"f_{i}", "", _KEN, _YEARS) for i, sv in enumerate(svs)],
        role_names={sv: RoleNames(donor=None, recipient="financingRecipient") for sv in svs},
    )
    return FakeCoverageService(index)


def _binding(prop: str, value: str) -> SlotBindingDraft:
    return SlotBindingDraft(axis="how", property_dcid=prop, kind="value", value_dcids=[value])


def _resolve(members: dict[str, dict], bindings: list[SlotBindingDraft]):
    return materialise(
        _shape(members),
        bindings,
        {"financingRecipient": [_KEN]},
        graph=FakeGraph(raise_on_call=True),
        sdmx=FakeSdmxClient(observations={}),
        coverage=_coverage(list(members)).snapshot(),
        discover_scope=True,
    )


def test_ppg_debt_service_serves_the_ppg_total_alone():
    result = _resolve(
        _MEMBERS,
        [_binding("financialFlowType", "ServicingPayment"), _binding("guaranteeStatus", _PPG)],
    )

    assert isinstance(result, Materialised)
    assert result.sv_dcids == [_PPG_TOTAL]
    assert result.population is not None
    assert set(result.population.admitted_dcids) == {_PPG_TOTAL, _PPG_PUBLIC_SECTOR}


def test_members_silent_on_every_bound_value_leave_binding_mismatch():
    silent = {sv: _MEMBERS[sv] for sv in (_BONDS_PUBLIC_SECTOR, _PRIVATE_GENERAL_GOVERNMENT)}

    result = _resolve(silent, [_binding("guaranteeStatus", _PPG)])

    assert isinstance(result, NoDataDraft)
    assert result.reason == "binding_mismatch"


def test_an_unbound_guarantee_status_still_serves_the_breakdowns():
    """Control: the rule reads bound properties only."""
    result = _resolve(_MEMBERS, [_binding("financialFlowType", "ServicingPayment")])

    assert isinstance(result, Materialised)
    assert _BONDS_PUBLIC_SECTOR in result.sv_dcids
    assert _PRIVATE_GENERAL_GOVERNMENT in result.sv_dcids
