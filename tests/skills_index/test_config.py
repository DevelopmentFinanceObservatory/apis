"""Tests attachment classification, skill id and dcid validation, and reference path validation."""

from __future__ import annotations

import pytest

from skills_index.config import (
    RESOURCE_PATH_MAX_LENGTH,
    SKILL_ID_GRAMMAR_MAX_LENGTH,
    classify_attachment,
    reference_path_rejection,
    skill_id_from_dcid,
)
from skills_index.types import (
    ConstraintPropertyAttachment,
    ConstraintValueAttachment,
    RejectedAttachment,
    TupleMemberAttachment,
)


def test_person_is_a_population_type_value():
    kind = classify_attachment(frozenset({"populationType", "domainIncludes"}))

    assert kind == TupleMemberAttachment(tuple_property="populationType")


def test_count_is_a_measured_property_value():
    kind = classify_attachment(frozenset({"measuredProperty"}))

    assert kind == TupleMemberAttachment(tuple_property="measuredProperty")


def test_measured_value_is_a_stat_type_value():
    kind = classify_attachment(frozenset({"statType"}))

    assert kind == TupleMemberAttachment(tuple_property="statType")


def test_nominal_is_a_measurement_qualifier_value():
    kind = classify_attachment(frozenset({"measurementQualifier"}))

    assert kind == TupleMemberAttachment(tuple_property="measurementQualifier")


def test_crs_purpose_is_a_constraint_property():
    kind = classify_attachment(frozenset({"constraintProperties"}))

    assert kind == ConstraintPropertyAttachment()


def test_crs_purpose_code_is_a_constraint_value():
    kind = classify_attachment(frozenset({"crsPurpose"}))

    assert kind == ConstraintValueAttachment(property="crsPurpose")


def test_health_care_function_is_rejected_absent():
    kind = classify_attachment(frozenset())

    assert isinstance(kind, RejectedAttachment)


def test_bare_usa_is_rejected_absent():
    # The graph resolves `country/USA`. The bare form yields empty arcs, like `healthCareFunction`.
    kind = classify_attachment(frozenset())

    assert isinstance(kind, RejectedAttachment)


def test_two_tuple_properties_incoming_is_rejected_as_ambiguous():
    kind = classify_attachment(frozenset({"populationType", "measuredProperty"}))

    assert isinstance(kind, RejectedAttachment)


@pytest.mark.parametrize(
    "properties",
    [
        pytest.param({"domainIncludes", "rangeIncludes", "typeOf"}, id="schema-arcs"),
        # The skill publisher's own arcs: `Provenance <-rangeIncludes` names
        # skillAppliesToProvenance on the dev graph.
        pytest.param({"skillAppliesToProvenance", "isPartOf"}, id="provenance-skill-arc"),
        pytest.param({"skillAppliesTo"}, id="ontology-skill-arc"),
        pytest.param({"agentSkillReferences"}, id="reference-arc"),
    ],
)
def test_only_schema_or_skill_publisher_arcs_incoming_is_rejected(properties):
    kind = classify_attachment(frozenset(properties))

    assert isinstance(kind, RejectedAttachment)


@pytest.mark.parametrize(
    ("dcid", "skill_id"),
    [
        ("skill/oda-concepts", "oda-concepts"),
        ("skill/a", "a"),
        ("skill/a1-b2-c3", "a1-b2-c3"),
        ("skill/" + "a" * SKILL_ID_GRAMMAR_MAX_LENGTH, "a" * SKILL_ID_GRAMMAR_MAX_LENGTH),
    ],
)
def test_skill_id_from_dcid_returns_the_id_of_a_well_formed_skill_dcid(dcid, skill_id):
    assert skill_id_from_dcid(dcid) == skill_id


@pytest.mark.parametrize(
    "dcid",
    [
        "oda-concepts",
        "dc/skill/oda-concepts",
        "Skill/oda-concepts",
        "skill/",
        "skill/Oda",
        "skill/oda_concepts",
        "skill/-oda",
        "skill/oda-",
        "skill/oda--concepts",
        "skill/oda/concepts",
        "skill/healthz",
        "skill/" + "a" * (SKILL_ID_GRAMMAR_MAX_LENGTH + 1),
        "skill/oda\n",
    ],
)
def test_skill_id_from_dcid_refuses_a_dcid_that_is_not_skill_plus_a_valid_id(dcid):
    assert skill_id_from_dcid(dcid) is None


@pytest.mark.parametrize(
    "path",
    [
        "references/core-oda.md",
        "references/a/b-c/d1.md",
        "references/" + "a" * (RESOURCE_PATH_MAX_LENGTH - len("references/.md")) + ".md",
    ],
)
def test_reference_path_rejection_admits_a_valid_path(path):
    assert reference_path_rejection(path) is None


@pytest.mark.parametrize(
    ("path", "reason_fragment"),
    [
        ("", "empty"),
        ("a" * (RESOURCE_PATH_MAX_LENGTH - len(".md") + 1) + ".md", "ceiling"),
        ("references\\core-oda.md", "backslash"),
        ("/references/core-oda.md", "leading slash"),
        ("references/core-oda.csv", "does not end in"),
        ("references/core-oda", "does not end in"),
        ("references/core-oda.md.md", "segment outside"),
        ("references/../core-oda.md", "'..'"),
        ("../core-oda.md", "'..'"),
        ("references/./core-oda.md", "segment outside"),
        ("references//core-oda.md", "segment outside"),
        ("references/.md", "segment outside"),
        ("references/Core-ODA.md", "segment outside"),
        ("references/core_oda.md", "segment outside"),
        ("references/core oda.md", "segment outside"),
        ("SKILL.md", "does not sit under 'references/'"),
        ("skill.md", "does not sit under 'references/'"),
        ("core-oda.md", "does not sit under 'references/'"),
        ("other/y.md", "does not sit under 'references/'"),
        ("references.md", "does not sit under 'references/'"),
    ],
)
def test_reference_path_rejection_names_why_a_path_is_invalid(path, reason_fragment):
    reason = reference_path_rejection(path)

    assert reason is not None
    assert reason_fragment.replace("leading slash", "starts with a slash") in reason
