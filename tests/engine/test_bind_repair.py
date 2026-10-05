"""Tests for binding repair, degenerate binding detection, and starved-axis restoration.

Synthetic properties isolate constraint matching. The final integration test checks that a
malformed binding refuses through `resolve_variable`.
"""

from __future__ import annotations

import asyncio

import pytest

from code_index.shared import get_index
from search.engine.bind import (
    DraftBindingEvidence,
    DraftBindingKind,
    SlotBindingDraft,
    _UniversalBindOutput,
)
from search.engine.bind_repair import (
    _label_named_in_text,
    apply_declared_defaults,
    declared_default_slots_named,
    defaulted_slots_named,
    detect_degenerate_bindings,
    repair_draft_bindings,
    restore_starved_axis_binding,
)
from search.engine.qualifier_match import QualifierMatch, QualifierTie
from search.engine.region import resolve_variable
from search.engine.region.codes import DEGENERATE_BINDING_DETECTED, STARVED_AXIS_BINDING_RESTORED
from search.engine.shape import ShapeDraft, shape_draft_from
from search.models import Axis
from tests.engine._fake_coverage import empty_coverage_snapshot
from tests.engine._fake_sdmx import FakeSdmxClient
from tests.fixtures import FakeGraph, make_resolver

_SV_A = "SV_A"
_SV_B = "SV_B"

_SV1 = "sv_1"
_SV2 = "sv_2"


def _arcs(**constraints: str) -> dict:
    return {
        "constraintProperties": {"nodes": [{"dcid": p} for p in constraints]},
        **{prop: {"nodes": [{"dcid": val}]} for prop, val in constraints.items()},
    }


def _binding(
    prop: str,
    value: str,
    *,
    evidence: DraftBindingEvidence = "mentioned",
    kind: DraftBindingKind = "value",
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid=prop,
        kind=kind,
        value_dcids=[value] if kind == "value" else [],
        evidence=evidence,
    )


def _taxonomy(**slots: list[str]) -> dict[str, list[str]]:
    """Build `slot_taxonomy` entries for the `how` axis."""
    return {f"how:{prop}": values for prop, values in slots.items()}


class TestNoOpWhenAlreadyRealizable:
    @pytest.mark.parametrize(
        ("bindings", "slot_taxonomy", "sv_arc_facts"),
        [
            pytest.param(
                [_binding("concept", "X", evidence="mentioned")],
                _taxonomy(concept=["X"]),
                {_SV1: _arcs(concept="X")},
                id="single-bound-property",
            ),
            pytest.param(
                [
                    _binding("concept", "X", evidence="mentioned"),
                    _binding("concept_alt", "Y", evidence="sole_option"),
                ],
                _taxonomy(concept=["X"], concept_alt=["Y"]),
                {_SV1: _arcs(concept="X", concept_alt="Y")},
                id="two-bound-properties",
            ),
        ],
    )
    def test_realizable_draft_is_left_unchanged(self, bindings, slot_taxonomy, sv_arc_facts):
        members = [_SV1]

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestSoleOptionBlockerDropped:
    """A sole-option spelling is dropped when no member shares it with the other binding."""

    def _members_and_facts(self):
        members = [_SV1, _SV2]
        sv_arc_facts = {
            _SV1: _arcs(concept="X"),
            _SV2: _arcs(concept_alt="Y"),
        }
        return members, sv_arc_facts

    def test_sole_option_binding_dropped_and_slot_kept_present_unbound(self):
        members, sv_arc_facts = self._members_and_facts()
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            _binding("concept_alt", "Y", evidence="sole_option"),
        ]
        # Only _SV2 asserts concept_alt, with value "Y".
        slot_taxonomy = _taxonomy(concept=["X"], concept_alt=["Y"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert len(repaired) == 2
        assert len(dropped) == 1
        assert dropped[0].property_dcid == "concept_alt"

        kept = next(b for b in repaired if b.property_dcid == "concept")
        assert kept.kind == "value"
        assert kept.value_dcids == ["X"]

        unbound = next(b for b in repaired if b.property_dcid == "concept_alt")
        assert unbound.kind == "unbound"
        assert unbound.value_dcids == []


class TestMentionedBlockersLeaveDraftUnchanged:
    """Keep mentioned bindings on disjoint members, even when their combination is unrealizable."""

    def test_no_sole_option_binding_present_leaves_draft_and_reports_nothing_dropped(self):
        members = [_SV1, _SV2]
        sv_arc_facts = {
            _SV1: _arcs(concept="X"),
            _SV2: _arcs(concept_alt="Y"),
        }
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            _binding("concept_alt", "Y", evidence="mentioned"),
        ]
        slot_taxonomy = _taxonomy(concept=["X"], concept_alt=["Y"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestEvidenceDefaultNeverFiresRepair:
    """Bindings that omit `evidence` default to `mentioned` and remain untouched."""

    def test_binding_built_without_explicit_evidence_is_never_dropped(self):
        members = [_SV1, _SV2]
        sv_arc_facts = {
            _SV1: _arcs(concept="X"),
            _SV2: _arcs(concept_alt="Y"),
        }
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="concept", kind="value", value_dcids=["X"]),
            SlotBindingDraft(
                axis="how", property_dcid="concept_alt", kind="value", value_dcids=["Y"]
            ),
        ]
        assert bindings[0].evidence == "mentioned"
        assert bindings[1].evidence == "mentioned"
        slot_taxonomy = _taxonomy(concept=["X"], concept_alt=["Y"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestThreeSpellingOnePass:
    """Both unmentioned singleton bindings drop together when they block the named value."""

    def test_both_sole_option_bindings_drop_together(self):
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept_a="X")}
        bindings = [
            _binding("concept_a", "X", evidence="mentioned"),
            _binding("concept_b", "Y", evidence="sole_option"),
            _binding("concept_c", "Z", evidence="sole_option"),
        ]
        # concept_b/concept_c genuinely single-member (mirrors nn-05's
        # medicalStatus/transmissionMode, each observed as exactly one value).
        slot_taxonomy = _taxonomy(concept_a=["X"], concept_b=["Y"], concept_c=["Z"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert {b.property_dcid for b in dropped} == {"concept_b", "concept_c"}
        by_prop = {b.property_dcid: b for b in repaired}
        assert by_prop["concept_a"].kind == "value"
        assert by_prop["concept_b"].kind == "unbound"
        assert by_prop["concept_c"].kind == "unbound"


class TestStillUnrealizableAfterDroppingLeavesDraftUnchanged:
    """An unrealizable mentioned binding prevents partial repair."""

    def test_mentioned_property_no_member_asserts_returns_original_draft(self):
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept_alt="Y")}
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            _binding("concept_alt", "Y", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(concept_alt=["Y"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestFailOpenValueFilterBranch:
    """Repair mirrors the resolver's fail-open filter when a bound value matches no members."""

    def test_contradicting_sole_member_is_realizable_via_fail_open_no_drop_needed(self):
        """A conflicting value fails open to the original member.
        Repair leaves the draft intact.
        """
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept="OTHER")}
        bindings = [_binding("concept", "X", evidence="mentioned")]
        slot_taxonomy = _taxonomy(concept=["OTHER"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestMislabeledSoleOptionNotDropped:
    """Repair trusts `sole_option` only when the server taxonomy confirms a singleton."""

    def test_two_member_taxonomy_sole_option_leaves_draft_unchanged(self):
        """The draft says `sole_option`, but the two-value taxonomy keeps the blocking binding."""
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept="X")}
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            _binding("concept_alt", "Y", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(concept=["X"], concept_alt=["Y", "Z"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestNn04AndNn05CardinalityMatchesFixtures:
    """The singleton cardinality check permits the real nn-04 and nn-05 repairs."""

    def test_nn04_style_single_member_age_slot_still_dropped(self):
        members = ["death_0years", "mortalityevent_infant"]
        sv_arc_facts = {
            "death_0years": _arcs(age="Years0"),
            "mortalityevent_infant": _arcs(ageGroupClassification="Infant"),
        }
        bindings = [
            _binding("ageGroupClassification", "Infant", evidence="mentioned"),
            _binding("age", "Years0", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(ageGroupClassification=["Infant"], age=["Years0"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert len(dropped) == 1
        assert dropped[0].property_dcid == "age"
        kept = {b.property_dcid: b for b in repaired}
        assert kept["ageGroupClassification"].kind == "value"
        assert kept["age"].kind == "unbound"

    def test_nn05_style_two_single_member_slots_dropped_together(self):
        # The Measles member and the deceased/person-to-person member assert disjoint axes.
        members = ["measles", "patient_deceased_ptp"]
        sv_arc_facts = {
            "measles": _arcs(medicalCondition="Measles"),
            "patient_deceased_ptp": _arcs(
                medicalStatus="PatientDeceased",
                transmissionMode="PersonToPersonTransmission",
            ),
        }
        bindings = [
            _binding("medicalCondition", "Measles", evidence="mentioned"),
            _binding("medicalStatus", "PatientDeceased", evidence="sole_option"),
            _binding("transmissionMode", "PersonToPersonTransmission", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(
            medicalCondition=["Measles"],
            medicalStatus=["PatientDeceased"],
            transmissionMode=["PersonToPersonTransmission"],
        )

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert {b.property_dcid for b in dropped} == {"medicalStatus", "transmissionMode"}
        kept = {b.property_dcid: b for b in repaired}
        assert kept["medicalCondition"].kind == "value"
        assert kept["medicalStatus"].kind == "unbound"
        assert kept["transmissionMode"].kind == "unbound"


class TestGuardNeverDropsToEmptyRemainder:
    """Dropping both singleton bindings would leave an empty, trivially realizable draft."""

    def test_two_disjoint_sole_option_axes_not_repaired(self):
        members = [_SV_A, _SV_B]
        sv_arc_facts = {
            _SV_A: _arcs(gender="Female"),
            _SV_B: _arcs(age="Adult"),
        }
        bindings = [
            _binding("gender", "Female", evidence="sole_option"),
            _binding("age", "Adult", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(gender=["Female"], age=["Adult"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []

    def test_asymmetric_partial_drop_still_repairs(self):
        """Dropping one eligible binding leaves a non-empty, realizable draft."""
        members = [_SV_A, _SV_B]
        sv_arc_facts = {
            _SV_A: _arcs(gender="Female"),
            _SV_B: _arcs(age="Adult"),
        }
        bindings = [
            _binding("gender", "Female", evidence="mentioned"),
            _binding("age", "Adult", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(gender=["Female"], age=["Adult"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert len(dropped) == 1
        assert dropped[0].property_dcid == "age"
        kept = {b.property_dcid: b for b in repaired}
        assert kept["gender"].kind == "value"
        assert kept["age"].kind == "unbound"


class TestTaxonomyGateTightening:
    """A `sole_option` draft is eligible only when taxonomy confirms its single value."""

    @pytest.mark.parametrize(
        "slot_taxonomy",
        [
            pytest.param(_taxonomy(concept=["X"]), id="property-absent"),
            pytest.param(
                _taxonomy(concept=["X"], concept_alt=["Z"]),
                id="singleton-disagrees-with-draft",
            ),
        ],
    )
    def test_unverified_singleton_is_not_dropped(self, slot_taxonomy):
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept="X")}
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            _binding("concept_alt", "Y", evidence="sole_option"),
        ]

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []

    def test_multi_value_sole_option_draft_not_dropped(self):
        """A sole_option binding that drafted more than one value_dcid cannot
        itself be a verified singleton, whatever the taxonomy says."""
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept="X")}
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            SlotBindingDraft(
                axis="how",
                property_dcid="concept_alt",
                kind="value",
                value_dcids=["Y", "Z"],
                evidence="sole_option",
            ),
        ]
        slot_taxonomy = _taxonomy(concept=["X"], concept_alt=["Y"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


class TestDetectDegenerateBindings:
    """Empty `value` and `set` bindings are reported without mutation, regardless of evidence."""

    def test_empty_value_dcids_kind_value_detected(self):
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="concept",
                kind="value",
                value_dcids=[],
                evidence="sole_option",
            )
        ]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert unchanged[0] is bindings[0], "detection must not copy or mutate the binding"
        assert len(degenerate) == 1
        assert degenerate[0].property_dcid == "concept"

    def test_empty_value_dcids_kind_set_detected(self):
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="concept",
                kind="set",
                value_dcids=[],
                evidence="sole_option",
            )
        ]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert len(degenerate) == 1
        assert degenerate[0].property_dcid == "concept"

    def test_empty_value_dcids_detected_regardless_of_evidence(self):
        """An empty `value_dcids` is malformed for either evidence value."""
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="concept", kind="value", value_dcids=[])
        ]
        assert bindings[0].evidence == "mentioned"

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert len(degenerate) == 1

    def test_kind_value_with_one_value_dcid_not_detected(self):
        bindings = [_binding("concept", "X", kind="value")]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert degenerate == []

    def test_kind_set_with_one_value_dcid_not_detected(self):
        """A one-element set is a valid constraint for `bound_values_from_bindings`."""
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="concept", kind="set", value_dcids=["X"])
        ]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert degenerate == []

    def test_kind_set_with_two_value_dcids_not_detected(self):
        bindings = [
            SlotBindingDraft(
                axis="how", property_dcid="concept", kind="set", value_dcids=["X", "Y"]
            )
        ]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert degenerate == []

    def test_kind_unbound_with_stray_value_dcids_not_detected(self):
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="concept", kind="unbound", value_dcids=[])
        ]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert degenerate == []

    def test_kind_absent_not_detected(self):
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="concept", kind="absent", value_dcids=[])
        ]

        unchanged, degenerate = detect_degenerate_bindings(bindings)

        assert unchanged == bindings
        assert degenerate == []


class TestPostDropAcceptanceRejectsFailOpen:
    """Repair rejects a remainder that realizes only because a bound value fails open."""

    def test_dropping_the_eligible_binding_would_unmask_fail_open_not_repaired(self):
        members = [_SV1]
        sv_arc_facts = {_SV1: _arcs(concept="CONTRADICT")}
        bindings = [
            _binding("concept", "X", evidence="mentioned"),
            _binding("concept_alt", "Y", evidence="sole_option"),
        ]
        slot_taxonomy = _taxonomy(concept=["CONTRADICT"], concept_alt=["Y"])

        repaired, dropped = repair_draft_bindings(bindings, members, sv_arc_facts, slot_taxonomy)

        assert repaired == bindings
        assert dropped == []


# ---------------------------------------------------------------------------
# restore_starved_axis_binding
# ---------------------------------------------------------------------------


def _gender_arcs(gender: str | None) -> dict:
    """Arcs for a Person/count member: `gender` is a real constraint value
    when given, or the member is silent on the axis entirely (an ungendered
    aggregate) when None."""
    if gender is None:
        return {"constraintProperties": {"nodes": []}}
    return {
        "constraintProperties": {"nodes": [{"dcid": "gender"}]},
        "gender": {"nodes": [{"dcid": gender}]},
    }


def _race_arcs(race: str | None) -> dict:
    if race is None:
        return {"constraintProperties": {"nodes": []}}
    return {
        "constraintProperties": {"nodes": [{"dcid": "race"}]},
        "race": {"nodes": [{"dcid": race}]},
    }


def _starved_shape(
    *,
    constraint_props: list[str],
    sv_arc_facts: dict[str, dict],
    prop_observed_values: dict[str, list[str]],
    slot_taxonomy: dict[str, list[str]] | None = None,
    place_available: bool | None = True,
    value_labels: dict[str, str] | None = None,
):
    """Build a Person/count shape with configurable inputs for starvation guards."""
    return shape_draft_from(
        shape_id="person_count_measuredvalue",
        label="Population",
        pop_type_dcid="Person",
        meas_prop_dcid="count",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=constraint_props,
        prop_labels={p: p for p in constraint_props},
        prop_observed_values=prop_observed_values,
        sv_arc_facts=sv_arc_facts,
        slot_taxonomy=slot_taxonomy if slot_taxonomy is not None else {},
        place_available=place_available,
        value_labels=value_labels,
        recalled=False,
    )


def _togo_shape(
    *,
    slot_taxonomy: dict[str, list[str]] | None = None,
    place_available: bool | None = True,
    value_labels: dict[str, str] | None = None,
):
    """Build an ungendered aggregate with Female and Male siblings."""
    sv_arc_facts = {
        "Count_Person": _gender_arcs(None),
        "Count_Person_Female": _gender_arcs("Female"),
        "Count_Person_Male": _gender_arcs("Male"),
    }
    return _starved_shape(
        constraint_props=["gender"],
        sv_arc_facts=sv_arc_facts,
        prop_observed_values={"gender": ["Female", "Male"]},
        slot_taxonomy=slot_taxonomy,
        place_available=place_available,
        value_labels=value_labels
        if value_labels is not None
        else {"Female": "Female", "Male": "Male"},
    )


def _kenya_race_shape(
    *,
    slot_taxonomy: dict[str, list[str]] | None = None,
    place_available: bool | None = True,
    value_labels: dict[str, str] | None = None,
):
    """Build a race-starved shape with WhiteAlone and BlackOrAfricanAmerican values."""
    sv_arc_facts = {
        "Count_Person": _race_arcs(None),
        "Count_Person_WhiteAlone": _race_arcs("WhiteAlone"),
        "Count_Person_BlackOrAfricanAmerican": _race_arcs("BlackOrAfricanAmerican"),
    }
    return _starved_shape(
        constraint_props=["race"],
        sv_arc_facts=sv_arc_facts,
        prop_observed_values={"race": ["WhiteAlone", "BlackOrAfricanAmerican"]},
        slot_taxonomy=slot_taxonomy,
        place_available=place_available,
        value_labels=value_labels
        if value_labels is not None
        else {"WhiteAlone": "White alone", "BlackOrAfricanAmerican": "Black or African American"},
    )


def _unbound_binding(prop: str, axis: Axis = "how") -> SlotBindingDraft:
    return SlotBindingDraft(axis=axis, property_dcid=prop, kind="unbound", value_dcids=[])


class TestRestoreStarvedAxisBindingFires:
    def test_query_naming_the_starved_value_restores_binding_and_taxonomy(self):
        shape = _togo_shape()
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert len(restored) == 1
        fired = repaired[0]
        assert fired.kind == "value"
        assert fired.evidence == "mentioned"
        assert fired.value_dcids == ["Female"]
        # slot_taxonomy is restored in place, on the same shape object.
        assert shape.slot_taxonomy["how:gender"] == ["Female", "Male"]


class TestLabelNamedInTextCodeSystemValue:
    """Code-system labels omit suffixes so plain query wording can match (issue #866)."""

    def test_plain_code_system_label_matches_a_realistic_query(self):
        assert _label_named_in_text("standard grant", "give me odf standard grant flows") is True

    def test_suffixed_graph_label_never_matches_the_same_query(self):
        """The code-system suffix blocks a match to the plain query phrase."""
        assert (
            _label_named_in_text("standard grant (CRS Finance)", "give me standard grant flows")
            is False
        )


def _crs_finance_type_arcs(finance_type: str | None) -> dict:
    """Build CRS-flow arcs with an optional finance-type constraint."""
    if finance_type is None:
        return {"constraintProperties": {"nodes": []}}
    return {
        "constraintProperties": {"nodes": [{"dcid": "financeType"}]},
        "financeType": {"nodes": [{"dcid": finance_type}]},
    }


def _crs_finance_shape(
    *,
    slot_taxonomy: dict[str, list[str]] | None = None,
    place_available: bool | None = True,
):
    """Build a CRS-flow shape with two code-index labels (issue #866)."""
    sv_arc_facts = {
        "CrsFlow": _crs_finance_type_arcs(None),
        "CrsFlow_StandardGrant": _crs_finance_type_arcs("CRSTypeOfFinance110"),
        "CrsFlow_StandardLoan": _crs_finance_type_arcs("CRSTypeOfFinance421"),
    }
    return _starved_shape(
        constraint_props=["financeType"],
        sv_arc_facts=sv_arc_facts,
        prop_observed_values={
            "financeType": ["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
        },
        slot_taxonomy=slot_taxonomy,
        place_available=place_available,
        value_labels={
            "CRSTypeOfFinance110": "standard grant",
            "CRSTypeOfFinance421": "standard loan",
        },
    )


class TestRestoreStarvedAxisBindingCrsVocabulary:
    """A starved code-system value restores from its clean index label (issue #866)."""

    def test_query_naming_the_crs_value_in_plain_form_restores_binding(self):
        shape = _crs_finance_shape()
        bindings = [_unbound_binding("financeType")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "health ODA as a standard grant to Kenya"
        )

        assert len(restored) == 1
        fired = repaired[0]
        assert fired.kind == "value"
        assert fired.evidence == "mentioned"
        assert fired.value_dcids == ["CRSTypeOfFinance110"]
        assert shape.slot_taxonomy["how:financeType"] == [
            "CRSTypeOfFinance110",
            "CRSTypeOfFinance421",
        ]


class TestRestoreStarvedAxisBindingKenyaNegative:
    """A starved axis stays open when none of its values appears in the query."""

    def test_query_naming_no_race_value_never_fires(self):
        shape = _kenya_race_shape()
        bindings = [_unbound_binding("race")]

        repaired, restored = restore_starved_axis_binding(bindings, shape, "population of kenya")

        assert restored == []
        assert repaired == bindings
        assert repaired[0].kind == "unbound"
        assert shape.slot_taxonomy.get("how:race") in (None, [])


class TestRestoreStarvedAxisBindingTokenBoundary:
    """Whole-word matching keeps `female` and `male` distinct."""

    def test_female_query_matches_only_female(self):
        shape = _togo_shape()
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert len(restored) == 1
        assert repaired[0].value_dcids == ["Female"]

    def test_male_query_matches_only_male(self):
        shape = _togo_shape()
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "male population of togo"
        )

        assert len(restored) == 1
        assert repaired[0].value_dcids == ["Male"]


class TestRestoreStarvedAxisBindingNegationGuard:
    """Negated or excluded mentions must not restore an affirmative binding."""

    @pytest.mark.parametrize(
        "query",
        [
            pytest.param("not female population of togo", id="leading-negator"),
            pytest.param("population of togo excluding female", id="exclusion-framing"),
            pytest.param("non-female population of togo", id="fused-non-prefix"),
            pytest.param("non female population of togo", id="spaced-non-prefix"),
            pytest.param("population of togo other than female", id="other-than"),
            pytest.param("population of togo, exclude female", id="exclude-verb"),
            pytest.param("population of togo, excludes female", id="excludes-verb"),
            pytest.param("population of togo rather than female", id="rather-than"),
        ],
    )
    def test_exclusion_framing_never_restores(self, query: str):
        shape = _togo_shape()
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(bindings, shape, query)

        assert restored == []
        assert repaired == bindings
        assert shape.slot_taxonomy.get("how:gender") in (None, [])

    def test_affirmative_query_still_fires(self):
        """An affirmative mention restores the binding."""
        shape = _togo_shape()
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert len(restored) == 1
        assert repaired[0].kind == "value"
        assert repaired[0].value_dcids == ["Female"]
        assert shape.slot_taxonomy["how:gender"] == ["Female", "Male"]


class TestRestoreStarvedAxisBindingAmbiguous:
    def test_query_naming_both_values_never_fires(self):
        shape = _togo_shape()
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "population by gender (female or male) in togo"
        )

        assert restored == []
        assert repaired == bindings
        assert shape.slot_taxonomy.get("how:gender") in (None, [])


class TestRestoreStarvedAxisBindingPlaceAvailableNone:
    def test_place_available_none_never_fires(self):
        """`None` leaves starvation unconfirmed, so the repair stays inactive."""
        shape = _togo_shape(place_available=None)
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert restored == []
        assert repaired == bindings


class TestRestoreStarvedAxisBindingTaxonomyNonEmpty:
    def test_non_empty_slot_taxonomy_never_fires(self):
        """A non-empty place-scoped taxonomy leaves the binder's choice intact."""
        shape = _togo_shape(slot_taxonomy={"how:gender": ["Female"]})
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert restored == []
        assert repaired == bindings
        assert shape.slot_taxonomy == {"how:gender": ["Female"]}


class TestRestoreStarvedAxisBindingPlaceAvailableFalse:
    def test_place_available_false_never_fires(self):
        """`False` marks the no-data path. An empty taxonomy marks starvation."""
        shape = _togo_shape(place_available=False)
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert restored == []
        assert repaired == bindings
        assert shape.slot_taxonomy.get("how:gender") in (None, [])


class TestRestoreStarvedAxisBindingAlreadyBound:
    def test_already_bound_value_slot_survives_untouched(self):
        """An existing value binding is resolved, so restoration leaves it unchanged."""
        shape = _togo_shape()
        bindings = [_binding("gender", "Female")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert restored == []
        assert repaired == bindings
        assert repaired[0].kind == "value"
        assert repaired[0].value_dcids == ["Female"]


class TestRestoreStarvedAxisBindingNoDisagreement:
    def test_single_observed_value_across_full_group_never_fires(self):
        """Every member that asserts gender agrees on Female, making it a fixed structural fact.
        The restore leaves the slot open even when the query names Female.
        """
        shape = _starved_shape(
            constraint_props=["gender"],
            sv_arc_facts={
                "Count_Person": _gender_arcs(None),
                "Count_Person_Female": _gender_arcs("Female"),
            },
            prop_observed_values={"gender": ["Female"]},
            value_labels={"Female": "Female"},
        )
        bindings = [_unbound_binding("gender")]

        repaired, restored = restore_starved_axis_binding(
            bindings, shape, "female population of togo"
        )

        assert restored == []
        assert repaired == bindings
        assert shape.slot_taxonomy.get("how:gender") in (None, [])


class TestRestoreStarvedAxisBindingRealizabilityVeto:
    """Restoring infant beside `Years0` leaves no member satisfying both bindings."""

    def test_split_spelling_restore_would_be_unrealizable_and_is_vetoed(self):
        sv_arc_facts = {
            "death_0years": _arcs(age="Years0"),
            "mortalityevent_infant": _arcs(ageGroupClassification="Infant"),
            # Two ageGroupClassification values mark a disagreement axis.
            "mortalityevent_adult": _arcs(ageGroupClassification="Adult"),
        }
        shape = _starved_shape(
            constraint_props=["age", "ageGroupClassification"],
            sv_arc_facts=sv_arc_facts,
            prop_observed_values={
                "age": ["Years0"],
                "ageGroupClassification": ["Infant", "Adult"],
            },
            slot_taxonomy={"how:age": ["Years0"]},
            value_labels={"Years0": "0 years", "Infant": "infant", "Adult": "adult"},
        )
        bindings = [
            _binding("age", "Years0", evidence="sole_option"),
            _unbound_binding("ageGroupClassification"),
        ]

        repaired, restored = restore_starved_axis_binding(bindings, shape, "infant mortality rate")

        assert restored == []
        assert repaired == bindings
        unbound = next(b for b in repaired if b.property_dcid == "ageGroupClassification")
        assert unbound.kind == "unbound"
        assert unbound.value_dcids == []
        assert shape.slot_taxonomy.get("how:ageGroupClassification") in (None, [])


class TestDetectDegenerateBindingsIntegrationStandard:
    """A degenerate gender binding refuses through the standard path, even when the Male-only member
    has data.
    """

    _MALE_SV = "UnemploymentRate_Person_Male"
    _RECIPIENT = "country/USA"
    _DETECT_QUERY = "unemployment rate"

    class _DegenerateGenderLLM:
        """Elects the sole (Male-only) candidate shape but drafts gender as
        kind="value" with an empty value_dcids."""

        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="gender",
                        kind="value",
                        value_dcids=[],
                    ),
                ],
                also_plausible=None,
                ask=None,
            ), None

    @classmethod
    def _gender_arcs(cls, gender: str) -> dict:
        return {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "unemploymentRate"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
            "constraintProperties": {"nodes": [{"dcid": "gender"}]},
            "gender": {"nodes": [{"dcid": gender}]},
        }

    @classmethod
    def _graph(cls, *, with_data: bool) -> FakeGraph:
        nodes = {
            cls._MALE_SV: {
                "label": "Unemployment rate (male)",
                "type": "StatisticalVariable",
                "arcs": cls._gender_arcs("Male"),
            },
            "country/USA": {"label": "United States", "type": "Country"},
        }
        obs = (
            {
                f"{cls._MALE_SV}|{cls._RECIPIENT}": [
                    {"earliestDate": "2020", "latestDate": "2020", "obsCount": 1}
                ]
            }
            if with_data
            else {}
        )
        return FakeGraph(
            nodes=nodes,
            obs=obs,
            detect={cls._DETECT_QUERY: {"svs": [cls._MALE_SV]}},
            resolve={"USA": cls._RECIPIENT},
        )

    def _run(self, graph: FakeGraph):
        return asyncio.run(
            resolve_variable(
                "unemployment rate by gender",
                entities=["USA"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="unemployment rate by gender in USA",
                pac=True,
                graph=graph,
                resolver=make_resolver(graph),
                llm=self._DegenerateGenderLLM(),
                base_steps=[],
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )

    def test_degenerate_gender_binding_refuses_before_materialise(self):
        result = self._run(self._graph(with_data=False))

        assert result.status == "no_data"
        assert result.no_data_reason == "variable_not_resolved"
        assert result.elected_stat_vars is None
        assert any(w.code == DEGENERATE_BINDING_DETECTED for w in result.warnings), (
            "the degenerate gender binding must be reported as a warning, not just silently refused"
        )
        assert "materialise" not in result.timing_by_step, (
            "the pipeline must refuse before the materialise step runs at all"
        )

    def test_degenerate_gender_binding_refuses_even_with_positive_data_for_broadened_member(self):
        """The Male-only member has real, positive data for the broadened
        (as-if-unbound) reading -- refusal must hold regardless, mirroring the
        dev-finance sibling's own sentinel-data guard."""
        result = self._run(self._graph(with_data=True))

        assert result.status == "no_data", (
            "a degenerate gender binding must refuse, never widen into a confident "
            f"answer for a broader gender reading; got status={result.status!r}"
        )
        assert result.no_data_reason == "variable_not_resolved"


class TestRestoreStarvedAxisBindingIntegrationStandard:
    """A starved gender axis restores from query wording, then returns a flagged no-data result."""

    _AGGREGATE_SV = "Count_Person"
    _FEMALE_SV = "Count_Person_Female"
    _MALE_SV = "Count_Person_Male"
    _RECIPIENT = "country/TGO"
    _DETECT_QUERY = "population"

    class _StarvedGenderLLM:
        """Elects the sole shape and drafts gender as kind="unbound" -- what
        the real bind() call drafts when the availability floor has already
        emptied gender's slot_taxonomy entry."""

        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="gender",
                        kind="unbound",
                        value_dcids=[],
                    ),
                ],
                also_plausible=None,
                ask=None,
            ), None

    class _StarvedGenderAbsentLLM:
        """Same starved-gender draft as _StarvedGenderLLM, but kind="absent"
        -- the binder's other starved-slot spelling guard 3 accepts -- to
        prove the disclosure warning names the real original kind rather
        than a hardcoded one."""

        async def generate_structured(self, *, prompt, system, schema):
            assert schema.__name__ == "_UniversalBindOutput"
            return _UniversalBindOutput(
                chosen_shape_index=0,
                bindings=[
                    SlotBindingDraft(
                        axis="how",
                        property_dcid="gender",
                        kind="absent",
                        value_dcids=[],
                    ),
                ],
                also_plausible=None,
                ask=None,
            ), None

    @staticmethod
    def _population_gender_arcs(gender: str | None) -> dict:
        """Build Person/count arcs with an optional gender constraint."""
        arcs = {
            "populationType": {"nodes": [{"dcid": "Person"}]},
            "measuredProperty": {"nodes": [{"dcid": "count"}]},
            "statType": {"nodes": [{"dcid": "measuredValue"}]},
        }
        arcs.update(_gender_arcs(gender))
        return arcs

    @classmethod
    def _graph(cls) -> FakeGraph:
        nodes = {
            "gender": {"label": "Gender"},
            "Female": {"label": "Female"},
            "Male": {"label": "Male"},
            cls._AGGREGATE_SV: {
                "label": "Population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs(None),
            },
            cls._FEMALE_SV: {
                "label": "Female population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs("Female"),
            },
            cls._MALE_SV: {
                "label": "Male population",
                "type": "StatisticalVariable",
                "arcs": cls._population_gender_arcs("Male"),
            },
            cls._RECIPIENT: {"label": "Togo", "type": "Country"},
        }
        obs = {f"{cls._AGGREGATE_SV}|{cls._RECIPIENT}": [{"obsCount": 137}]}
        return FakeGraph(
            nodes=nodes,
            obs=obs,
            detect={cls._DETECT_QUERY: {"svs": [cls._AGGREGATE_SV, cls._FEMALE_SV, cls._MALE_SV]}},
            resolve={"Togo": cls._RECIPIENT},
        )

    def test_female_query_flags_the_gendered_member_as_no_data(self):
        result = asyncio.run(
            resolve_variable(
                "female population of togo",
                entities=["Togo"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="female population of togo",
                pac=True,
                graph=self._graph(),
                resolver=make_resolver(self._graph()),
                llm=self._StarvedGenderLLM(),
                base_steps=[],
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )

        assert result.status == "no_data"
        assert result.no_data_reason == "no_observations"
        assert any(w.code == STARVED_AXIS_BINDING_RESTORED for w in result.warnings), (
            "the restored gender binding must be reported as a warning"
        )
        assert result.flagged_match is not None
        assert len(result.flagged_match) == 1
        flagged = result.flagged_match[0]
        assert result.gate == "no_member_has_data_at_place"
        assert [sv.ref for sv in flagged.resolved.stat_vars.items] == [self._FEMALE_SV], (
            "the flagged match must name the gendered member the restored "
            "binding pointed at, never the ungendered aggregate"
        )

    def test_absent_origin_restore_warning_names_the_real_kind(self):
        """The warning reports `absent`, the original kind before restoration."""
        result = asyncio.run(
            resolve_variable(
                "female population of togo",
                entities=["Togo"],
                date_request=None,
                detect_query=self._DETECT_QUERY,
                role_query="female population of togo",
                pac=True,
                graph=self._graph(),
                resolver=make_resolver(self._graph()),
                llm=self._StarvedGenderAbsentLLM(),
                base_steps=[],
                base_timing={},
                sdmx=FakeSdmxClient(observations={}, availability={}),
                coverage=empty_coverage_snapshot(),
            )
        )

        restored_warning = next(
            w for w in result.warnings if w.code == STARVED_AXIS_BINDING_RESTORED
        )
        assert restored_warning.message == (
            "'gender' was drafted as kind='absent' because the availability "
            "floor emptied its slot_taxonomy at the queried place; the "
            "variable phrase names one of its unscoped values, so the "
            "binding is restored to that value."
        )


# ---------------------------------------------------------------------------
# apply_declared_defaults
# ---------------------------------------------------------------------------


def _grants_shape(
    *,
    crs_type_of_flow_taxonomy: list[str] | None = ("CRSTypeOfFlow10", "CRSTypeOfFlow21"),
) -> ShapeDraft:
    """Build a CRS-flow shape with explicit slot taxonomies for default tests."""
    taxonomy: dict[str, list[str]] = {
        "how:crsTypeOfFinance": ["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
    }
    if crs_type_of_flow_taxonomy is not None:
        taxonomy["how:crsTypeOfFlow"] = list(crs_type_of_flow_taxonomy)
    return shape_draft_from(
        shape_id="crs_flow_amount",
        label="CRS flow",
        pop_type_dcid="FinancialFlow",
        meas_prop_dcid="amount",
        stat_type_dcid="measuredValue",
        meas_qual_dcid=None,
        meas_denom_dcid=None,
        constraint_props=["crsTypeOfFinance", "crsTypeOfFlow"],
        prop_labels={"crsTypeOfFinance": "instrument", "crsTypeOfFlow": "flow"},
        prop_observed_values={
            "crsTypeOfFinance": ["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
            "crsTypeOfFlow": list(crs_type_of_flow_taxonomy or ()),
        },
        sv_arc_facts={},
        slot_taxonomy=taxonomy,
        place_available=True,
        value_labels=None,
        recalled=False,
    )


def _flow_unbound(evidence: DraftBindingEvidence = "mentioned") -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how", property_dcid="crsTypeOfFlow", kind="unbound", evidence=evidence
    )


def _finance(
    value: str = "CRSTypeOfFinance110",
    *,
    kind: DraftBindingKind = "value",
    evidence: DraftBindingEvidence = "mentioned",
) -> SlotBindingDraft:
    return SlotBindingDraft(
        axis="how",
        property_dcid="crsTypeOfFinance",
        kind=kind,
        value_dcids=[value] if kind == "value" else [],
        evidence=evidence,
    )


class TestApplyDeclaredDefaultsFires:
    def test_grants_with_no_flow_type_named_defaults_to_oda(self):
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]

        repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert len(defaulted) == 1
        assert defaulted[0].property_dcid == "crsTypeOfFlow"
        assert defaulted[0].kind == "unbound"
        flow = next(b for b in repaired if b.property_dcid == "crsTypeOfFlow")
        assert flow.kind == "value"
        assert flow.value_dcids == ["CRSTypeOfFlow10"]
        # crsTypeOfFinance is untouched.
        finance = next(b for b in repaired if b.property_dcid == "crsTypeOfFinance")
        assert finance.value_dcids == ["CRSTypeOfFinance110"]

    def test_fires_through_a_descendant_of_the_grant_row_not_only_the_row_itself(self):
        """The real descendant CRSTypeOfFinance210 triggers the grant default through the index."""
        shape = _grants_shape()
        bindings = [_finance("CRSTypeOfFinance210"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert len(defaulted) == 1


class TestApplyDeclaredDefaultsDoesNotFire:
    def test_shape_declares_no_crs_type_of_flow_slot(self):
        # The shape has no crsTypeOfFlow slot.
        shape = shape_draft_from(
            shape_id="crs_finance_only",
            label="CRS finance",
            pop_type_dcid="FinancialFlow",
            meas_prop_dcid="amount",
            stat_type_dcid="measuredValue",
            meas_qual_dcid=None,
            meas_denom_dcid=None,
            constraint_props=["crsTypeOfFinance"],
            prop_labels={"crsTypeOfFinance": "instrument"},
            prop_observed_values={"crsTypeOfFinance": ["CRSTypeOfFinance110"]},
            sv_arc_facts={},
            slot_taxonomy={"how:crsTypeOfFinance": ["CRSTypeOfFinance110"]},
            place_available=True,
            value_labels=None,
            recalled=False,
        )
        bindings = [_finance()]

        repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []
        assert repaired == bindings

    def test_crs_type_of_flow_already_bound(self):
        shape = _grants_shape()
        bindings = [
            _finance(),
            SlotBindingDraft(
                axis="how",
                property_dcid="crsTypeOfFlow",
                kind="value",
                value_dcids=["CRSTypeOfFlow21"],
            ),
        ]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_evidence_requested_all_suppresses_the_default(self):
        """A requested-all flow slot stays open even when a grant instrument is bound."""
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound(evidence="requested_all")]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_a_flow_type_match_recognised_in_text_suppresses_the_default(self):
        """A recognised flow-type match suppresses the default even when its binding is unbound."""
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]
        match = QualifierMatch(
            code_system="crsTypeOfFlow",
            code_dcid="CRSTypeOfFlow21",
            label="non-export credit official flow",
            source="synonym",
            matched_text="other official flows",
        )

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), query_qualifier_matches=(match,)
        )

        assert defaulted == []

    def test_a_flow_type_tie_recognised_in_text_suppresses_the_default(self):
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]
        tie = QualifierTie(
            matched_text="flows",
            rows=(get_index().codes_by_normalized_text("official development assistance")[0],),
        )

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), query_qualifier_ties=(tie,)
        )

        assert defaulted == []

    def test_finance_bound_but_not_mentioned_does_not_fire(self):
        """The grants default requires a text-derived finance binding."""
        shape = _grants_shape()
        bindings = [_finance(evidence="sole_option"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_bound_to_a_loan_value_does_not_fire(self):
        """The grants default applies to finance values in the grant subtree."""
        shape = _grants_shape()
        bindings = [_finance("CRSTypeOfFinance421"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_itself_unbound_does_not_fire(self):
        shape = _grants_shape()
        bindings = [
            SlotBindingDraft(axis="how", property_dcid="crsTypeOfFinance", kind="unbound"),
            _flow_unbound(),
        ]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_shape_never_carries_the_oda_flow_type(self):
        """The default is skipped when CRSTypeOfFlow10 is absent from the shape taxonomy."""
        shape = _grants_shape(crs_type_of_flow_taxonomy=["CRSTypeOfFlow21", "CRSTypeOfFlow60"])
        bindings = [_finance(), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_bound_as_a_set_does_not_fire(self):
        """A mixed set leaves the grants default inactive."""
        shape = _grants_shape()
        bindings = [
            SlotBindingDraft(
                axis="how",
                property_dcid="crsTypeOfFinance",
                kind="set",
                value_dcids=["CRSTypeOfFinance110", "CRSTypeOfFinance421"],
                evidence="mentioned",
            ),
            _flow_unbound(),
        ]

        _repaired, defaulted = apply_declared_defaults(bindings, shape, index=get_index())

        assert defaulted == []

    def test_finance_trigger_engine_authored_does_not_fire(self):
        """Query-derived defaults require text-authored finance bindings."""
        shape = _grants_shape()
        bindings = [_finance(), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            engine_authored_properties=frozenset({"crsTypeOfFinance"}),
        )

        assert defaulted == []


class TestApplyDeclaredDefaultsWhenNamed:
    """The whole-word term "aid" can trigger the default in the variable or raw query."""

    def test_aid_as_the_whole_variable_fires(self):
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable="aid to Kenya"
        )

        assert len(defaulted) == 1
        assert defaulted[0].property_dcid == "crsTypeOfFlow"
        flow = next(b for b in repaired if b.property_dcid == "crsTypeOfFlow")
        assert flow.kind == "value"
        assert flow.value_dcids == ["CRSTypeOfFlow10"]

    def test_aid_in_query_text_alone_fires(self):
        """The raw query triggers the default when extraction omits "aid"."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            variable="funding to Kenya",
            query_text="How much aid did Kenya get?",
        )

        assert [b.property_dcid for b in defaulted] == ["crsTypeOfFlow"]

    def test_negated_aid_in_query_text_does_not_fire(self):
        """The raw query uses the same negation check as the extracted variable."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            variable="funding to Kenya",
            query_text="non-aid funding to Kenya",
        )

        assert defaulted == []

    def test_aid_inside_another_word_does_not_fire(self):
        """Whole-word matching distinguishes "aid" from "paid" and "aided"."""
        shape = _grants_shape()
        for variable in ("who paid Kenya in 2020", "France aided Kenya in 2020"):
            bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

            _repaired, defaulted = apply_declared_defaults(
                bindings, shape, index=get_index(), variable=variable
            )

            assert defaulted == [], variable

    def test_a_named_flow_type_blocks_the_named_trigger(self):
        """A flow-type match suppresses the default even when the variable names "aid"."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]
        match = QualifierMatch(
            code_system="crsTypeOfFlow",
            code_dcid="CRSTypeOfFlow21",
            label="non-export credit official flow",
            source="synonym",
            matched_text="other official flows",
        )

        _repaired, defaulted = apply_declared_defaults(
            bindings,
            shape,
            index=get_index(),
            variable="aid",
            query_qualifier_matches=(match,),
        )

        assert defaulted == []

    def test_a_requested_all_flow_slot_blocks_the_named_trigger(self):
        """A requested-all flow slot stays open despite the "aid" trigger."""
        shape = _grants_shape()
        bindings = [
            _finance(kind="unbound", evidence="mentioned"),
            _flow_unbound(evidence="requested_all"),
        ]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable="aid to Kenya across all flows"
        )

        assert defaulted == []

    @pytest.mark.parametrize(
        "variable",
        [
            pytest.param("not aid financing to kenya", id="leading-negator"),
            pytest.param("financing to kenya excluding aid", id="exclusion-framing"),
            pytest.param("non-aid financing to kenya", id="fused-non-prefix"),
            pytest.param("non aid financing to kenya", id="spaced-non-prefix"),
            pytest.param("financing to kenya other than aid", id="other-than"),
            pytest.param("financing to kenya, exclude aid", id="exclude-verb"),
            pytest.param("financing to kenya, excludes aid", id="excludes-verb"),
            pytest.param("financing to kenya rather than aid", id="rather-than"),
        ],
    )
    def test_negated_or_excluded_aid_does_not_fire(self, variable: str):
        """Negated or excluded "aid" must not trigger the ODA default."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable=variable
        )

        assert defaulted == [], variable

    def test_a_bound_flow_slot_blocks_the_named_trigger(self):
        """An existing CRSTypeOfFlow30 binding takes precedence over the "aid" trigger."""
        shape = _grants_shape()
        bindings = [
            _finance(kind="unbound", evidence="mentioned"),
            SlotBindingDraft(
                axis="how",
                property_dcid="crsTypeOfFlow",
                kind="value",
                value_dcids=["CRSTypeOfFlow30"],
            ),
        ]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable="private aid to Kenya"
        )

        assert defaulted == []

    @pytest.mark.parametrize(
        "variable",
        [
            pytest.param("aid-for-trade flows to Kenya", id="hyphenated-compound"),
            pytest.param("aid for trade to Kenya", id="spaced-compound"),
        ],
    )
    def test_aid_for_trade_fires_the_oda_default(self, variable: str):
        """Aid for Trade is an OECD/DAC category within ODA, with or without a hyphen."""
        shape = _grants_shape()
        bindings = [_finance(kind="unbound", evidence="mentioned"), _flow_unbound()]

        _repaired, defaulted = apply_declared_defaults(
            bindings, shape, index=get_index(), variable=variable
        )

        assert len(defaulted) == 1
        assert defaulted[0].property_dcid == "crsTypeOfFlow"


class TestDefaultedSlotsNamed:
    """A fired default is reported only where a text names its slot's plain name."""

    @pytest.mark.parametrize(
        ("texts", "expected"),
        [
            pytest.param(
                ["aid flows"],
                [("flow", "official development assistance")],
                id="plural-in-variable",
            ),
            pytest.param(
                ["aid", "top 5 flow"], [("flow", "official development assistance")], id="singular"
            ),
            pytest.param(["aid", "Which sectors got the most aid"], [], id="no-flow-noun"),
            pytest.param(["aid", "cashflow"], [], id="not-a-whole-word"),
        ],
    )
    def test_reports_the_slot_and_its_default_label(self, texts, expected):
        _repaired, defaulted = apply_declared_defaults(
            [_finance(kind="unbound"), _flow_unbound()],
            _grants_shape(),
            index=get_index(),
            variable="aid",
        )

        assert defaulted_slots_named(defaulted, index=get_index(), texts=texts) == expected

    def test_declared_default_slots_named_reads_only_slots_with_a_default(self):
        """Every declared plain name appears, but only the flow slot has a default."""
        text = "aid flows by sector, instrument and modality"

        assert declared_default_slots_named(text, index=get_index()) == ("crsTypeOfFlow",)
