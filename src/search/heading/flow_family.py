"""The flow family of an elected shape, read from its members' constraint values.

A family decides how a heading names the roles of a role-bearing series
(`dfo_platform.roles.role_wording`): "Germany's gross ODA disbursements to Kenya" for
aid, "Ghana's external debt service owed to China" for debt. Role names cannot decide it,
because OECD CRS and World Bank IDS StatVars both declare `financingProvider` and
`financingRecipient`. The population type and the flow or position type can.

The rules read exact values the dev graph declares:

- Every OECD CRS and ONE imputed StatVar in the code index (22,208) declares
  `crsTypeOfFlow`, on a `FinancialFlow` or `FinancialObligation` population. No IDS
  StatVar declares it, so it alone tells CRS disbursements from IDS disbursements, which
  share `financialFlowType = Disbursement`.
- IDS flows are `FinancialFlow` with `financialFlowType` `ServicingPayment`,
  `InterestPayment`, `PrincipalRepayment` or `Disbursement`. IDS stocks are
  `FinancialPosition` with `financialPositionType` `AmountOutstanding`,
  `PrincipalArrears`, `InterestArrears` or `AmountUndisbursed`. A total and its
  by-creditor twin carry the same values, so both read in one family.

The function needs only members' constraint values, so it runs at bind time on a
`ShapeDraft`'s arcs as well as after materialise on the served members.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from dfo_platform.roles import FlowFamily

_AID_POPULATIONS: Final = frozenset({"FinancialFlow", "FinancialObligation"})
_AID_MARKER: Final = "crsTypeOfFlow"

_FLOW_TYPE: Final = "financialFlowType"
_POSITION_TYPE: Final = "financialPositionType"

_DEBT_FLOW_FAMILIES: Final[Mapping[str, FlowFamily]] = {
    "ServicingPayment": "debt_service",
    "InterestPayment": "debt_service",
    "PrincipalRepayment": "debt_service",
    "Disbursement": "debt_lending",
}
"""`financialFlowType` values on a `FinancialFlow` StatVar without `crsTypeOfFlow`."""

_DEBT_POSITION_FAMILIES: Final[Mapping[str, FlowFamily]] = {
    "AmountOutstanding": "debt_stock",
    "PrincipalArrears": "debt_stock",
    "InterestArrears": "debt_stock",
    "AmountUndisbursed": "debt_lending",
}
"""`financialPositionType` values on a `FinancialPosition` StatVar."""


def _member_family(
    population_type: str | None, constraints: Mapping[str, str]
) -> FlowFamily | None:
    if population_type in _AID_POPULATIONS and _AID_MARKER in constraints:
        return "aid"
    if population_type == "FinancialFlow":
        return _DEBT_FLOW_FAMILIES.get(constraints.get(_FLOW_TYPE, ""))
    if population_type == "FinancialPosition":
        return _DEBT_POSITION_FAMILIES.get(constraints.get(_POSITION_TYPE, ""))
    return None


def flow_family(
    *, population_type: str | None, members: Iterable[Mapping[str, str]]
) -> FlowFamily | None:
    """The flow family every member of a shape shares, or `None`.

    Args:
        population_type: the shape's `populationType` dcid (`ShapeDraft.pop_type_dcid`,
            `Shape.population_type.dcid`).
        members: each member's constraint values, property dcid to value dcid
            (`MemberFacts.read_constraints()`, `StatVarRow.constraint_properties`).

    Returns:
        The shared family. `None` when there are no members, when a member falls in no
        family, or when members fall in different families (debt service beside debt
        disbursements). A renderer reads `None` as `GENERIC_ROLE_WORDING`.
    """
    families = {_member_family(population_type, constraints) for constraints in members}
    if len(families) != 1:
        return None
    return families.pop()
