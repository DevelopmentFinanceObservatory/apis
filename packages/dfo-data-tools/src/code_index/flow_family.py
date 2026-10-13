"""The flow family of an elected shape, read from its members.

A family decides how a heading names the roles of a role-bearing series
(`dfo_platform.roles.role_wording`): "Germany's gross ODA disbursements to Kenya" for
aid, "Ghana's external debt service owed to China" for debt. Role names cannot decide it,
because OECD aid StatVars and World Bank IDS StatVars both declare `financingProvider` and
`financingRecipient`.

A StatVar the code index holds takes its family from the account its datasets declare
(`code_index.types.DatasetDeclaration.account`): the provider's account reads as aid. The
datasets are those of the index family the row's provenances assign it
(`CodeIndex.family_by_dcid`).

A StatVar outside the index is read from its population type and constraint values, which
the dev graph declares for World Bank IDS series:

- IDS flows are `FinancialFlow` with `financialFlowType` `ServicingPayment`,
  `InterestPayment`, `PrincipalRepayment` or `Disbursement`.
- IDS stocks are `FinancialPosition` with `financialPositionType` `AmountOutstanding`,
  `PrincipalArrears`, `InterestArrears` or `AmountUndisbursed`.

A total and its by-creditor twin carry the same values, so both read in one family.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from code_index.query import CodeIndex
from dfo_platform.roles import FlowAccount, FlowFamily

_ACCOUNT_FAMILIES: Final[Mapping[FlowAccount, FlowFamily]] = {"provider": "aid"}
"""The family of a StatVar whose datasets declare each account."""

_FLOW_TYPE: Final = "financialFlowType"
_POSITION_TYPE: Final = "financialPositionType"

_DEBT_FLOW_FAMILIES: Final[Mapping[str, FlowFamily]] = {
    "ServicingPayment": "debt_service",
    "InterestPayment": "debt_service",
    "PrincipalRepayment": "debt_service",
    "Disbursement": "debt_lending",
}
"""`financialFlowType` values on a `FinancialFlow` StatVar outside the index."""

_DEBT_POSITION_FAMILIES: Final[Mapping[str, FlowFamily]] = {
    "AmountOutstanding": "debt_stock",
    "PrincipalArrears": "debt_stock",
    "InterestArrears": "debt_stock",
    "AmountUndisbursed": "debt_lending",
}
"""`financialPositionType` values on a `FinancialPosition` StatVar."""


def _declared_account(index: CodeIndex, dcid: str) -> FlowAccount | None:
    """The account the datasets of `dcid`'s index family declare, or `None` for a StatVar
    the index does not hold or whose datasets declare different accounts."""
    family = index.family_by_dcid.get(dcid)
    if family is None:
        return None
    accounts = {
        index.datasets[provenance].account
        for provenance in family.datasets
        if provenance in index.datasets
    }
    return accounts.pop() if len(accounts) == 1 else None


def _member_family(
    population_type: str | None, constraints: Mapping[str, str]
) -> FlowFamily | None:
    if population_type == "FinancialFlow":
        return _DEBT_FLOW_FAMILIES.get(constraints.get(_FLOW_TYPE, ""))
    if population_type == "FinancialPosition":
        return _DEBT_POSITION_FAMILIES.get(constraints.get(_POSITION_TYPE, ""))
    return None


def flow_family(
    *,
    index: CodeIndex,
    population_type: str | None,
    stat_vars: Iterable[str],
    members: Iterable[Mapping[str, str]],
) -> FlowFamily | None:
    """The flow family every member of a shape shares, or `None`.

    Args:
        index: the code index whose rows and dataset declarations decide the family.
        population_type: the shape's `populationType` dcid (`Shape.population_type.dcid`).
        stat_vars: the members' StatVar dcids. When one is an index row, the index
            decides the family.
        members: each member's constraint values, property dcid to value dcid
            (`StatVar.constraint_properties`), read when no StatVar is an index row.

    Returns:
        The shared family. `None` when there are no members, when a member falls in no
        family, when members fall in different families (debt service beside debt
        disbursements), or when an index row sits beside a StatVar the index does not
        hold or whose account differs. A renderer reads `None` as
        `GENERIC_ROLE_WORDING`.
    """
    accounts = {_declared_account(index, dcid) for dcid in stat_vars}
    if declared := {account for account in accounts if account is not None}:
        return _ACCOUNT_FAMILIES[declared.pop()] if len(accounts) == 1 else None
    families = {_member_family(population_type, constraints) for constraints in members}
    if len(families) != 1:
        return None
    return families.pop()
