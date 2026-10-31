"""Every option a live axis rule emits names its subject for a reader, since the label is
the option's only reader copy ("Counted as: health, CRS 120 alone · Africa's countries
only")."""

from __future__ import annotations

import pytest

from code_index.shared import get_index
from search.models import BaseIdentity, EntityGroup
from search.scope.rules import AXIS_RULES
from search.scope.tables import ScopeFacts
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.scope.test_region import _LISTED, _RANK, _SUMMED
from tests.scope.test_region import _facts as _region_facts
from tests.scope.test_sector import _HEALTH, _base
from tests.scope.test_sector import _facts as _sector_facts

# A donor's DAC1 total ODA in grant equivalents, as the parsed door resolves it.
_DAC1_TOTAL = BaseIdentity.model_validate(
    {
        "identity_version": "spec1",
        "shape": {
            "kind": "statvar_schema",
            "population_type": "FinancialFlow",
            "measured_property": "grantEquivalent",
            "stat_type": "measuredValue",
            "measurement_qualifier": "Nominal",
        },
        "slots": [
            {"property": "dac1AidType", "binding": {"state": "value", "ref": "DAC1AidType1010"}},
            {"property": "financialFlowType", "binding": {"state": "value", "ref": "Disbursement"}},
        ],
        "entities": {"origin": ["country/USA"]},
        "requested_window": {"kind": "range", "start_year": 2019, "end_year": 2019},
    }
)

_CASES = {
    "health-grouping": (_base({"state": "set", "refs": list(_HEALTH)}), _sector_facts()),
    "basic-health": (_base({"state": "value", "ref": "CRSPurposeCode122"}), _sector_facts()),
    "two-codes": (
        _base({"state": "set", "refs": ["CRSPurposeCode110", "CRSPurposeCode122"]}),
        _sector_facts(),
    ),
    "africa-summed": (_SUMMED, _region_facts()),
    "africa-ranked": (_LISTED, _region_facts(operations=(_RANK,))),
    "africa-no-group": (_SUMMED, _region_facts(group=None)),
    "asia-registry-group": (
        _LISTED.model_copy(
            update={"entities": {"origin": ["country/FRA"], "destination": ["country/IDN"]}}
        ),
        _region_facts(
            group=EntityGroup(label="Asia (DAC)", group_id="groups/OECD.DAC.Region.Asia")
        ),
    ),
    "dac1-total": (
        _DAC1_TOTAL,
        ScopeFacts(index=get_index(), coverage=empty_coverage_snapshot()),
    ),
}


def _labels(case: str) -> dict[str, str]:
    base, facts = _CASES[case]
    return {
        option.option_id: option.label
        for rule in AXIS_RULES.values()
        if (table := rule(base, None, facts)) is not None
        for option in table.options
    }


@pytest.mark.parametrize("case", list(_CASES))
def test_every_option_a_rule_emits_has_a_label_other_than_its_id(case):
    labels = _labels(case)

    assert labels
    assert all(label.strip() and label != option_id for option_id, label in labels.items())


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            "health-grouping",
            {
                "sector:120+130": "health incl. population and reproductive health (CRS 120 + 130)",
                "sector:120": "health, CRS 120 alone",
            },
        ),
        ("two-codes", {"sector:110+122": "education (CRS 110) + basic health (CRS 122)"}),
        (
            "africa-summed",
            {
                "region:members_and_residuals": "Africa incl. regional and unspecified amounts",
                "region:members_only": "Africa's countries only",
            },
        ),
        # The region is named by its total's declared name whatever group is posted, so a
        # search and its switch name it alike.
        ("africa-no-group", {"region:members_only": "Africa's countries only"}),
        ("asia-registry-group", {"region:members_only": "Asia's 47 countries (UN regions)"}),
    ],
)
def test_a_label_names_the_codes_or_the_region_it_counts(case, expected):
    labels = _labels(case)

    assert {option_id: labels[option_id] for option_id in expected} == expected


def test_a_dac1_total_offers_every_provider_scope_with_a_label():
    """The labels themselves are checked by
    `test_every_option_a_rule_emits_has_a_label_other_than_its_id`."""
    provider = {
        option_id for option_id in _labels("dac1-total") if option_id.startswith("provider:")
    }

    assert provider == {
        "provider:bilateral",
        "provider:inclusive",
        "provider:contributions",
        "provider:multilateral",
    }
