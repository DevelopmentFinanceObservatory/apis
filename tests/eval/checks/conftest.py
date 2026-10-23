"""Recorded responses, a wire schema, and case builders for the check tests.

Responses come from the sentence panel golden (`tests/fixtures/sentence_panel`), which
keeps each part's status, warnings and specs but drops the envelope; `panel` restores the
root `kind` and `status` the checks read. The schema is the one `GET /api/search/schema`
serves. A Spec's served StatVars are the members the golden stores beside it
(`sentence_facts`), which `panel_members` answers as the members door would.
"""

from __future__ import annotations

import copy
import functools
import gzip
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from scripts.sentence_golden import spec_dicts
from search.eval.cases import Case, Family, Gl, validate_case
from search.eval.checks import ScoreContext, WireSchema
from search.eval.checks.base import MembersRefused
from search.models import RESOLVE_RESPONSE_ADAPTER

PANEL_DIR = Path(__file__).parents[2] / "fixtures" / "sentence_panel"

IDS = "provenance/InternationalDebtStatistics"

FAMILIES = {
    "crs.disbursement_gross": Family(
        origin="dfo",
        dataset="provenance/OECDCreditorReportingSystem",
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
    ),
    # crs.disbursement_gross's shape and slots with StatVar pins, or under another dataset:
    # the "unserved" pins name a StatVar crs-032 does not serve, "served" one it does, and
    # the unconfirmed family's slots are unconfirmed.
    "crs.disbursement_unserved": Family(
        origin="dfo",
        dataset="provenance/OECDCreditorReportingSystem",
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
        stat_vars=["Amount_Debt_Disbursement"],
    ),
    "ids.disbursement_served": Family(
        origin="dfo",
        dataset=IDS,
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
        stat_vars=[
            "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow21_CRSTypeOfFinance421"
            "_CRSPurposeCode23230_Nominal"
        ],
    ),
    "ids.disbursement_unserved": Family(
        origin="dfo",
        dataset=IDS,
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
        stat_vars=["Amount_Debt_Disbursement"],
    ),
    "ids.disbursement_unconfirmed": Family(
        origin="dfo",
        dataset=IDS,
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed=Gl(gl="the disbursement slots"),
    ),
    "crs.flow": Family(
        origin="dfo",
        dataset="provenance/OECDCreditorReportingSystem",
        shape={"population_type": "FinancialFlow"},
    ),
    "crs.disbursement_pin_unconfirmed": Family(
        origin="dfo",
        dataset="provenance/OECDCreditorReportingSystem",
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
        stat_vars=[Gl(gl="the disbursement StatVar")],
    ),
    "crs.disbursement_gl_dataset": Family(
        origin="dfo",
        dataset=Gl(gl="the disbursement dataset"),
        shape={"population_type": "FinancialFlow", "measurement_qualifier": "Nominal"},
        fixed={"financialFlowType": "Disbursement", "nettingBasis": "Gross"},
    ),
    "crs.commitment": Family(
        origin="dfo",
        dataset="provenance/OECDCreditorReportingSystem",
        shape={"population_type": "FinancialObligation"},
        fixed={"financialObligationType": "Commitment"},
    ),
    "wdi.gdp": Family(
        origin="base",
        dataset="dc/base/WorldDevelopmentIndicators",
        shape={"population_type": "EconomicActivity", "measurement_denominator": None},
        fixed={"activitySource": "GrossDomesticProduction"},
    ),
    "wdi.gdp_per_capita": Family(
        origin="base",
        dataset="dc/base/WorldDevelopmentIndicators",
        shape={"population_type": "EconomicActivity", "measurement_denominator": "PerCapita"},
        fixed={"activitySource": "GrossDomesticProduction"},
    ),
    "weo.gdp_per_capita": Family(
        origin="dfo",
        dataset="provenance/IMFWorldEconomicOutlook",
        shape={"population_type": "EconomicProduction", "measurement_denominator": "Count_Person"},
    ),
    "weo.gdp_any": Family(
        origin="dfo",
        dataset="provenance/IMFWorldEconomicOutlook",
        shape={"population_type": "EconomicProduction"},
    ),
}


@functools.cache
def _panel_rows() -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(PANEL_DIR.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


# The served StatVars of Specs a test builds beside the recorded ones, by spec id.
_BUILT_MEMBERS: dict[str, list[str]] = {}


@functools.cache
def _recorded_members() -> dict[str, list[str]]:
    return {
        spec["spec_id"]: [member["ref"] for member in spec["sentence_facts"]["members"]]
        for row in _panel_rows().values()
        for part in row["parts"]
        for spec in spec_dicts(part)
    }


def serve(spec_id: str, members: list[str]) -> None:
    """Have `panel_members` answer `spec_id`, a Spec a test built, with `members`."""
    _BUILT_MEMBERS[spec_id] = members


def panel_members(spec_id: str, definition: dict[str, Any]) -> list[str]:
    """The StatVars a recorded Spec serves, or those `serve` gave a built one: a members
    door over the golden."""
    return _BUILT_MEMBERS.get(spec_id) or _recorded_members()[spec_id]


def panel(row_id: str) -> dict[str, Any]:
    """The recorded response for a sentence-panel query, with its root restored."""
    parts = copy.deepcopy(_panel_rows()[row_id]["parts"])
    statuses = {part["status"] for part in parts}
    return {
        "kind": "resolution",
        "status": statuses.pop() if len(statuses) == 1 else "mixed",
        "parts": parts,
        "diagnostics": {"warnings": []},
    }


def make_case(expect: dict[str, Any], *, door: str = "raw_text") -> Case:
    raw: dict[str, Any] = {"id": "t-1", "query": "q", "why": "test", "door": door}
    if door == "parsed":
        raw["parsed"] = [{"variable_text": "aid"}]
    return validate_case({**raw, "expect": expect}, families=FAMILIES)


def response_schema() -> dict[str, Any]:
    return RESOLVE_RESPONSE_ADAPTER.json_schema(mode="serialization")


def schema_listing_members() -> WireSchema:
    """The current response schema with an `items` array added to `ResolvedStatVars`, as an
    app built before the members door lists the StatVars a Spec serves. The current wire
    carries `member_count` and `served_count` there and no list."""
    schema = copy.deepcopy(response_schema())
    schema["$defs"]["ResolvedStatVars"]["properties"]["items"] = {
        "type": "array",
        "items": {"type": "object", "properties": {"ref": {"type": "string"}}},
    }
    return WireSchema.from_document({"response": schema})


def no_members_door(spec_id: str, definition: dict[str, Any]) -> list[str]:
    """The members door of an app built before it: every read answers 404."""
    request = httpx.Request("POST", f"http://base/api/datasets/{spec_id}/members")
    raise httpx.HTTPStatusError("404", request=request, response=httpx.Response(404))


def refusing_members_door(spec_id: str, definition: dict[str, Any]) -> list[str]:
    """A members door that refuses every definition as one it will not mint."""
    raise MembersRefused(code="member_not_resolvable", detail=f"{spec_id} did not resolve")


@pytest.fixture
def ctx() -> ScoreContext:
    return ScoreContext(
        schema=WireSchema.from_document({"response": response_schema()}), members=panel_members
    )
