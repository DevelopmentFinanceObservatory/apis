"""Test a stored warning's structured fields and their absence elsewhere."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest

from assembler.mint import warned_provenance_refs, warned_refs
from dataset_store.snapshot import ManifestFacts, StoredVersion
from datasets_api.manifest import (
    stored_manifest,
    version_document,
    warning_model,
)
from datasets_api.manifest_models import WarningModel
from datasets_api.origin import _withheld_warning

from ._version_facts import facts_from_body, golden_cases


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
        pytest.param(
            {
                "code": "withheld_by_registry",
                "severity": "warn",
                "message": "m",
                "unserved_entities": {"count": 1, "refs": ["dcid:country/PSE"]},
                "withheld_members": {"count": 2, "refs": ["dcid:MemberA", "dcid:MemberB"]},
                "withheld_provenances": {"count": 1, "refs": ["dcid:OECD_DAC2A"]},
            },
            id="withheld-by-registry",
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


_WITHHELD = {
    "code": "withheld_by_registry",
    "severity": "warn",
    "message": "m",
    "unserved_entities": {"count": 1, "refs": ["dcid:country/PSE"]},
    "withheld_members": {"count": 2, "refs": ["dcid:MemberA", "dcid:MemberB"]},
    "withheld_provenances": {"count": 1, "refs": ["dcid:dc/base/WHO_GHO"]},
}


def _stored_with(warning: Mapping[str, object]) -> StoredVersion:
    """A golden dataset version holding only `warning`, as storage returns it."""
    case = next(case for case in golden_cases().values() if "manifest" in case["expected"])
    facts = facts_from_body(case["body"])
    assert isinstance(facts, ManifestFacts)
    return StoredVersion(
        subject_id=case["subject_id"],
        subject_kind="spec",
        content_version=case["content_version"],
        schema_version=1,
        facts=replace(facts, warnings=[warning]),
        base_content_version=case["base_content_version"],
    )


@pytest.mark.parametrize(
    ("warning", "refs", "provenance_refs"),
    [
        pytest.param(
            _WITHHELD,
            ["dcid:country/PSE", "dcid:MemberA", "dcid:MemberB"],
            ["dcid:dc/base/WHO_GHO"],
            id="withheld-fields",
        ),
        # A row stored before the fields existed names its places and provenances in its
        # message alone.
        pytest.param(
            _withheld_warning("Count_Person", {"country/PSE": ("dc/base/WHO_GHO",)}).model_dump(
                mode="json"
            ),
            [],
            [],
            id="stored-before-the-fields",
        ),
    ],
)
def test_a_stored_withheld_warning_reads_back_and_names_its_refs(warning, refs, provenance_refs):
    """A stored version's warning dumps as recorded and its refs reach the frame's label read,
    entities and series apart from provenances."""
    stored = _stored_with(warning)

    assert stored_manifest(stored).model_dump(mode="json")["warnings"] == [warning]
    document = version_document(stored)
    assert warned_refs(manifest=document) == refs
    assert warned_provenance_refs(manifest=document) == provenance_refs
