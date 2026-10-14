"""Test candidate assembly, ordering, truncation, and summary sentences."""

from __future__ import annotations

from search.engine.assemble import build_candidates_part
from search.models import CandidatesPart
from tests.engine._assemble_fixtures import _make_spec


def test_build_candidates_part_returns_candidates_part():
    """Build a candidates response part."""
    spec_health = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    spec_malaria = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
    )

    part = build_candidates_part(
        [spec_health, spec_malaria],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    assert part.status == "candidates"


def test_assemble_candidates_has_two_or_more_distinct_spec_ids():
    """Candidate spec ids are unique."""
    spec_health = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    spec_malaria = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
    )

    part = build_candidates_part(
        [spec_health, spec_malaria],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    assert len(specs) >= 2
    spec_ids = [s.spec_id for s in specs]
    # All spec_ids must be pairwise distinct
    assert len(spec_ids) == len(set(spec_ids)), f"Duplicate spec_ids found: {spec_ids}"


def test_assemble_candidates_keeps_the_order_given_as_intent_rank():
    """The status rule gives the elected reading first and its rivals in menu order, and
    assembly keeps that order whatever the member counts, under `intent_rank`."""
    spec_a = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        member_count=5,
    )
    spec_b = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
        member_count=1,
    )
    spec_c = _make_spec(
        sv_dcid="Count_Person_Employed",
        purpose_dcid="DAC/Education",
        place_dcid="country/ETH",
        member_count=3,
    )

    part = build_candidates_part(
        [spec_b, spec_c, spec_a], variable_text="test variable", warnings=[]
    )

    assert isinstance(part, CandidatesPart)
    assert [s.spec_id for s in part.candidates.specs] == [
        spec_b.spec_id,
        spec_c.spec_id,
        spec_a.spec_id,
    ]
    assert part.candidates.ordering == "intent_rank"


def test_assemble_candidates_states_the_ordering_a_resubmit_posted():
    spec_a = _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH")
    spec_b = _make_spec(
        sv_dcid="Count_Household", purpose_dcid="DAC/Malariacontrol", place_dcid="country/ETH"
    )

    part = build_candidates_part(
        [spec_a, spec_b], variable_text="test variable", warnings=[], ordering="broadest_first"
    )

    assert isinstance(part, CandidatesPart)
    assert part.candidates.ordering == "broadest_first"
    assert [s.spec_id for s in part.candidates.specs] == [spec_a.spec_id, spec_b.spec_id]


# CandidateSet.truncated


def test_assemble_candidates_truncated_false_default_config():
    """The default cap marks an uncut list as complete."""
    spec_a = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    spec_b = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
    )

    part = build_candidates_part(
        [spec_a, spec_b],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    assert part.candidates.truncated is False


def test_assemble_candidates_truncated_true_when_max_candidates_cuts():
    """Clamping three specs to two sets truncated while respecting the two-spec minimum."""
    spec_a = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        member_count=1,
    )
    spec_b = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
        member_count=3,
    )
    spec_c = _make_spec(
        sv_dcid="Count_Person_Employed",
        purpose_dcid="DAC/Education",
        place_dcid="country/ETH",
        member_count=2,
    )

    part = build_candidates_part(
        [spec_a, spec_b, spec_c],
        variable_text="test variable",
        warnings=[],
        max_candidates=2,
    )

    assert isinstance(part, CandidatesPart)
    assert len(part.candidates.specs) == 2
    assert part.candidates.truncated is True


def test_assemble_candidates_truncated_true_when_budget_truncated_flag_set():
    """An upstream candidate-budget cut marks the response truncated."""
    spec_a = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    spec_b = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
    )

    part = build_candidates_part(
        [spec_a, spec_b],
        variable_text="test variable",
        warnings=[],
        budget_truncated=True,
    )

    assert isinstance(part, CandidatesPart)
    assert part.candidates.truncated is True


def test_assemble_candidates_sentence_truncated_with_known_total():
    """The count gives the known total and says the first two are shown, then states
    each shown reading."""
    spec_a = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        member_count=1,
    )
    spec_b = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
        member_count=3,
    )
    spec_c = _make_spec(
        sv_dcid="Count_Person_Employed",
        purpose_dcid="DAC/Education",
        place_dcid="country/ETH",
        member_count=2,
    )

    part = build_candidates_part(
        [spec_a, spec_b, spec_c],
        variable_text="test variable",
        warnings=[],
        max_candidates=2,
    )

    assert isinstance(part, CandidatesPart)
    assert len(part.candidates.specs) == 2
    assert part.rendered_sentence == (
        "3 readings; the first 2 are shown. "
        "ETH's ODA Grants for health. ETH's ODA Grants for malariacontrol."
    )


def test_assemble_candidates_sentence_truncated_with_unknown_total():
    """An unknown total produces a count that reports only the shown readings."""
    spec_a = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    spec_b = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
    )

    part = build_candidates_part(
        [spec_a, spec_b],
        variable_text="test variable",
        warnings=[],
        budget_truncated=True,
    )

    assert isinstance(part, CandidatesPart)
    assert part.rendered_sentence == (
        "More than 2 readings; the first 2 are shown. "
        "ETH's ODA Grants for health. ETH's ODA Grants for malariacontrol."
    )


# The candidates part's sentence


def test_assemble_candidates_sentence_states_each_reading():
    """The sentence counts the readings, then states each one, with the bound purpose
    that sets it apart."""
    spec_health = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    spec_malaria = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
    )

    part = build_candidates_part(
        [spec_health, spec_malaria],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    assert len(part.candidates.specs) == 2
    assert part.rendered_sentence == (
        "2 readings. ETH's ODA Grants for health. ETH's ODA Grants for malariacontrol."
    )
