"""Test a stored `stat_vars_summed` warning's structured fields and their absence elsewhere."""

from __future__ import annotations

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
