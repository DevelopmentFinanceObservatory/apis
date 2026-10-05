"""Recorded responses, a wire schema, and case builders for the check tests.

Responses come from the sentence panel golden (`tests/fixtures/sentence_panel`), which
keeps each part's status, warnings and specs but drops the envelope; `panel` restores the
root `kind` and `status` the checks read. The schema is the one `GET /api/search/schema`
serves.
"""

from __future__ import annotations

import copy
import functools
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from search.eval.cases import Case, Family, Gl, validate_case
from search.eval.checks import ScoreContext, WireSchema
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
    # the "unserved" pins name a StatVar crs-001 does not serve, "served" one it does, and
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
            "Amount_FinancialFlow_Gross_Disbursement_CRSTypeOfFlow10_CRSTypeOfFinance110"
            "_CRSPurposeCode11120_Nominal"
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


@pytest.fixture
def ctx() -> ScoreContext:
    return ScoreContext(schema=WireSchema.from_document({"response": response_schema()}))
