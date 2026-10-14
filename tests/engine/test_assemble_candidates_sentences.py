"""Test per-candidate sentence plans and the how-slot distinction guard."""

from __future__ import annotations

from search.engine.assemble import build_candidates_part
from search.models import CandidatesPart
from search.render.tokens import sentence_from_plan
from tests.engine._assemble_fixtures import (
    _make_promoting_spec,
    _make_spec,
)

# Per-candidate sentence plans.


def test_assemble_candidates_populates_sentences():
    """Each candidate gets a distinct sentence matching its plan."""
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
    assert len(specs) == 2
    for s in specs:
        assert s.resolved.rendered_sentence == sentence_from_plan(s.resolved.sentence_plan)
    assert specs[0].resolved.rendered_sentence != specs[1].resolved.rendered_sentence


def test_assemble_candidates_sentences_align_with_order_and_clamp():
    """Sentences follow the given order of the clamped candidate list."""
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
        member_count=5,
    )
    spec_c = _make_spec(
        sv_dcid="Count_Person_Employed",
        purpose_dcid="DAC/Education",
        place_dcid="country/ETH",
        member_count=3,
    )

    # The order given stands whatever the member counts.
    part = build_candidates_part(
        [spec_a, spec_b, spec_c],
        variable_text="test variable",
        warnings=[],
        max_candidates=2,
    )

    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    # The first two specs given survive the cap.
    assert [s.spec_id for s in specs] == [spec_a.spec_id, spec_b.spec_id]
    for s in specs:
        assert s.resolved.rendered_sentence is not None


def test_no_candidate_sentence_carries_a_note():
    """A candidate's own sentence states its reading, and no departure note."""
    strong = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        member_count=2,
    )
    loose = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
        member_count=3,
    )

    part = build_candidates_part([strong, loose], variable_text="test variable", warnings=[])

    assert isinstance(part, CandidatesPart)
    assert [spec.spec_id for spec in part.candidates.specs] == [strong.spec_id, loose.spec_id]
    for spec in part.candidates.specs:
        assert not any(token.kind == "note" for token in spec.resolved.sentence_plan or [])
    assert part.departures == []


# Preserve how slots when they distinguish candidate sentences.


def test_assemble_candidates_keeps_how_property_that_distinguishes_candidates():
    """Keep a differing how value in both sentences."""
    spec_health = _make_promoting_spec(purpose_dcid="Health", place_dcid="country/ETH")
    spec_education = _make_promoting_spec(purpose_dcid="Education", place_dcid="country/ETH")

    part = build_candidates_part(
        [spec_health, spec_education],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    assert len(specs) == 2
    for s in specs:
        how_tok = next(t for t in s.resolved.sentence_plan if t.axis == "how")
        assert how_tok.text != ""
    assert specs[0].resolved.rendered_sentence != specs[1].resolved.rendered_sentence


def test_assemble_candidates_folds_how_property_that_matches_across_candidates():
    """Fold a shared how value when the where value distinguishes the candidates."""
    spec_ethiopia = _make_promoting_spec(purpose_dcid="Health", place_dcid="country/ETH")
    spec_kenya = _make_promoting_spec(purpose_dcid="Health", place_dcid="country/KEN")

    part = build_candidates_part(
        [spec_ethiopia, spec_kenya],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    assert len(specs) == 2
    for s in specs:
        how_tok = next(t for t in s.resolved.sentence_plan if t.axis == "how")
        assert how_tok.text == ""


def test_assemble_candidates_keeps_boolean_how_property_that_distinguishes_candidates():
    """Keep differing boolean how values visible in both sentences."""
    spec_true = _make_promoting_spec(purpose_dcid="True", place_dcid="country/ETH")
    spec_false = _make_promoting_spec(purpose_dcid="False", place_dcid="country/ETH")

    part = build_candidates_part(
        [spec_true, spec_false],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    assert len(specs) == 2
    for s in specs:
        assert s.resolved.sentence_plan is not None
        how_tok = next(t for t in s.resolved.sentence_plan if t.axis == "how")
        assert how_tok.text != ""
    assert specs[0].resolved.rendered_sentence != specs[1].resolved.rendered_sentence


def test_assemble_candidates_keeps_how_property_bound_on_one_sibling_unbound_on_another():
    """A bound/unbound how-slot difference keeps the sentences distinct."""
    spec_bound = _make_promoting_spec(purpose_dcid="Health", place_dcid="country/ETH")
    spec_unbound = _make_promoting_spec(purpose_dcid=None, place_dcid="country/ETH")

    part = build_candidates_part(
        [spec_bound, spec_unbound],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    specs = part.candidates.specs
    assert len(specs) == 2
    assert specs[0].resolved.rendered_sentence != specs[1].resolved.rendered_sentence


def test_assemble_candidates_kept_property_already_named_still_folds_without_duplication():
    """Omit a value already named in what-text. Render the differing unnamed value as how."""
    spec_named = _make_promoting_spec(purpose_dcid="GDP", place_dcid="country/ETH")
    spec_clause = _make_promoting_spec(purpose_dcid="Health", place_dcid="country/ETH")

    part = build_candidates_part(
        [spec_named, spec_clause],
        variable_text="test variable",
        warnings=[],
    )

    assert isinstance(part, CandidatesPart)
    by_id = {s.spec_id: s for s in part.candidates.specs}
    named = by_id[spec_named.spec_id]
    clause = by_id[spec_clause.spec_id]

    assert named.resolved.sentence_plan is not None
    named_how_tok = next(t for t in named.resolved.sentence_plan if t.axis == "how")
    assert named_how_tok.text == ""
    assert named.resolved.rendered_sentence is not None
    assert named.resolved.rendered_sentence.lower().count("gdp") == 1

    assert clause.resolved.sentence_plan is not None
    clause_how_tok = next(t for t in clause.resolved.sentence_plan if t.axis == "how")
    assert clause_how_tok.text != ""

    assert named.resolved.rendered_sentence != clause.resolved.rendered_sentence
