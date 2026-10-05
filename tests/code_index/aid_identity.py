"""The education-aid-to-Kenya identity and the facts the packaged index holds for it.

The Datasets mint tests and the Search tests resolve this one identity.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

EDUCATION_AID_SPEC_ID = "spec_5a427ccbf716df03efcd8896d335dee4"

EDUCATION_AID_IDENTITY: dict = {
    "identity_version": "spec1",
    "shape": {
        "kind": "statvar_schema",
        "population_type": "FinancialFlow",
        "measured_property": "amount",
        "stat_type": "measuredValue",
        "measurement_qualifier": "Nominal",
    },
    "slots": [
        {"property": "crsCooperationModality", "binding": {"state": "unbound"}},
        {"property": "crsPurpose", "binding": {"state": "value", "ref": "CRSPurposeCode110"}},
        {"property": "crsTypeOfFinance", "binding": {"state": "unbound"}},
        {"property": "crsTypeOfFlow", "binding": {"state": "value", "ref": "CRSTypeOfFlow10"}},
        {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}},
        {"property": "nettingBasis", "binding": {"state": "value", "ref": "Gross"}},
    ],
    "entities": {"destination": ["country/KEN"]},
    "requested_window": {"kind": "all"},
}

AID_TYPE = "dac2aAidType"
IMPUTED_MULTILATERAL = "DAC2aAidType106"

CRS_PROVENANCE = "provenance/OECDCreditorReportingSystem"
IMPUTED_PROVENANCE = "provenance/ONEImputedMultilateralSectors"

# CRS members of the education family in the packaged index, and those among them that
# declare no `crsCooperationModality`.
CRS_MEMBER_COUNT = 257
CRS_MEMBERS_WITHOUT_MODALITY = 22

# Imputed multilateral members of the education family in the packaged index, all on the
# `attributedAmount` five-tuple.
IMPUTED_MEMBER_COUNT = 16

# The StatVars a Search response listed for the identity when a family spanned both
# datasets, as recorded from the deployed service, and the imputed members among them,
# whose `Amount_` dcids are absent from the packaged index.
LISTED_STAT_VARS: tuple[str, ...] = tuple(
    json.loads(
        (Path(__file__).parent / "fixtures" / "education_aid_listed_stat_vars.json").read_text()
    )
)
LISTED_IMPUTED_COUNT = 13


def education_aid_identity(*, aid_type: str | None = None, source: str | None = None) -> dict:
    """A copy of the identity, with an aid-type slot bound to `aid_type` and the provenance
    facet bound to `source` when given. Either binding changes the identity's hash. CRS rows
    declare no aid type, so `aid_type` spells an identity that predates the imputed
    table's own `attributedAmount` five-tuple."""
    identity = copy.deepcopy(EDUCATION_AID_IDENTITY)
    if aid_type is not None:
        identity["slots"].append(
            {"property": AID_TYPE, "binding": {"state": "value", "ref": aid_type}}
        )
        identity["slots"].sort(key=lambda slot: slot["property"])
    if source is not None:
        identity["facets"] = {"provenance": {"state": "value", "ref": source}}
    return identity


def imputed_education_aid_identity(*, source: str | None = None) -> dict:
    """The imputed multilateral counterpart of the identity: the same purpose, flow type and
    measure bindings on the `attributedAmount` five-tuple, with the slots its rows declare and
    the aid-type slot bound to the one value they hold. `source` binds the provenance facet."""
    identity = education_aid_identity(aid_type=IMPUTED_MULTILATERAL, source=source)
    identity["shape"]["measured_property"] = "attributedAmount"
    identity["slots"] = [
        slot
        for slot in identity["slots"]
        if slot["property"] not in ("crsCooperationModality", "crsTypeOfFinance")
    ]
    return identity
