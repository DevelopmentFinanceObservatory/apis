"""Test a stored warning's structured fields and their absence elsewhere."""

from __future__ import annotations

import pytest

from datasets_api.manifest import warning_model
from datasets_api.manifest_models import WarningModel


def _summed(*, label: str | None) -> dict[str, object]:
    return {
        "code": "stat_vars_summed",
        "severity": "info",
        "message": "",
        "contributing_member_count": 74,
        "stat_var_ref": "dcid:spec_" + "a" * 32,
        "stat_var_label": label,
    }


def test_a_warning_of_another_code_dumps_only_code_severity_and_message():
    """Stored version bodies dump without `exclude_none`, so the stored shape of every other
    warning stays exactly `{code, severity, message}`."""
    warning = WarningModel(code="revision_unavailable", severity="info", message="m")
    assert warning.model_dump(mode="json") == {
        "code": "revision_unavailable",
        "severity": "info",
        "message": "m",
    }


def test_a_stored_stat_vars_summed_warning_keeps_its_count_ref_and_label():
    model = WarningModel.model_validate(_summed(label="Education, ODA Grants"))
    assert model.contributing_member_count == 74
    assert model.stat_var_ref == "dcid:spec_" + "a" * 32
    assert model.stat_var_label == "Education, ODA Grants"


def test_a_stored_stat_vars_summed_warning_without_a_label_omits_the_label_key():
    dumped = WarningModel.model_validate(_summed(label=None)).model_dump(mode="json")
    assert "stat_var_label" not in dumped
    assert dumped["contributing_member_count"] == 74


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(
            {
                "code": "coverage_below_scope",
                "severity": "warn",
                "message": "2 of 9 series report at a different frequency and are left out "
                "of the figures.",
                "member_count": 9,
                "off_cadence_members": {"count": 2, "refs": ["dcid:MemberA", "dcid:MemberB"]},
                "out_of_window_members": {"count": 1, "refs": ["dcid:MemberC"]},
                "unserved_members": {"count": 1, "refs": ["dcid:MemberD"]},
            },
            id="group-members",
        ),
        pytest.param(
            {
                "code": "coverage_below_scope",
                "severity": "warn",
                "message": "No observations were served for 6 requested entities.",
                "unserved_entities": {
                    "count": 6,
                    "refs": [f"dcid:country/C{i}" for i in range(5)],
                },
            },
            id="unserved-entities",
        ),
        pytest.param(
            {
                "code": "coverage_below_scope",
                "severity": "warn",
                "message": "Requested up to 2030; the data covers 2020 to 2024.",
                "requested_years": {"end": 2030},
                "served_years": {"start": 2020, "end": 2024},
            },
            id="window-clamp-years",
        ),
        pytest.param(
            {
                "code": "currency_unmapped_omitted",
                "severity": "warn",
                "message": "It left out 1 entity.",
                "omitted_entities": {"count": 1, "refs": ["dcid:country/PSE"]},
            },
            id="currency-omitted-entities",
        ),
        pytest.param(
            {
                "code": "deflator_missing_omitted",
                "severity": "warn",
                "message": "It left out rows of 2 entities.",
                "omitted_entity_years": {
                    "count": 2,
                    "entities": [
                        {"ref": "dcid:country/AZE", "years": ["2022"]},
                        {"ref": "dcid:country/TWN", "years": ["1990-2001", "2004"]},
                    ],
                },
            },
            id="deflator-omitted-entities",
        ),
    ],
)
def test_a_stored_warning_with_counted_refs_reads_back_unchanged(stored):
    """A stored version body whose warning carries counted refs reads back, so a revision
    that does not write them still serves versions that do."""
    assert WarningModel.model_validate(stored).model_dump(mode="json") == stored


def test_a_coverage_warning_keeps_the_members_it_names_through_the_manifest():
    """A group's `coverage_below_scope` reaches the stored version with its member fields, and
    a field naming no member stays out of the dump."""
    warning = {
        "code": "coverage_below_scope",
        "severity": "warn",
        "message": "m",
        "member_count": 9,
        "off_cadence_members": {"count": 6, "refs": [f"dcid:Member{i}" for i in range(5)]},
    }

    dumped = warning_model(warning).model_dump(mode="json")

    assert dumped == warning
