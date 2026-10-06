"""Test candidate assembly, ordering, truncation, and summary sentences."""

from __future__ import annotations

import itertools

from search.engine.assemble import build_candidates_part
from search.engine.region.codes import NEAR_NEIGHBOR_AMBIGUOUS
from search.models import CandidatesPart, Warning
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


def test_assemble_candidates_ordering_is_broadest_first():
    """The default path orders candidates broadest first."""
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
    assert part.candidates.ordering == "broadest_first"


def test_assemble_candidates_bind_ranked_overrides_broadest_first():
    """Preserve bind order even when member counts rank the candidates differently."""
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
        [spec_b, spec_c, spec_a],
        variable_text="test variable",
        warnings=[],
        bind_ranked=True,
    )

    assert isinstance(part, CandidatesPart)
    ordered_ids = [s.spec_id for s in part.candidates.specs]
    assert ordered_ids == [spec_b.spec_id, spec_c.spec_id, spec_a.spec_id]


def test_assemble_candidates_moves_loose_readings_after_the_base_rank():
    """Place weak and near-neighbour readings after stronger ones."""
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
        member_count=4,
    )
    spec_c = _make_spec(
        sv_dcid="Count_Person_Employed",
        purpose_dcid="DAC/Education",
        place_dcid="country/ETH",
        member_count=3,
    )
    spec_d = _make_spec(
        sv_dcid="Count_Employment",
        purpose_dcid="DAC/Agriculture",
        place_dcid="country/ETH",
        member_count=2,
    )

    part = build_candidates_part(
        [spec_d, spec_c, spec_b, spec_a],
        variable_text="test variable",
        warnings=[],
        candidate_scores={spec_a.spec_id: 0.7},
        near_neighbor_spec_ids=frozenset({spec_b.spec_id}),
    )

    assert isinstance(part, CandidatesPart)
    assert [spec.spec_id for spec in part.candidates.specs] == [
        spec_c.spec_id,
        spec_d.spec_id,
        spec_a.spec_id,
        spec_b.spec_id,
    ]


def test_assemble_candidates_clamps_after_moving_loose_readings_last():
    """The cap keeps stronger candidates ahead of loose broad matches."""
    weak = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
        member_count=3,
    )
    strong_first = _make_spec(
        sv_dcid="Count_Household",
        purpose_dcid="DAC/Malariacontrol",
        place_dcid="country/ETH",
        member_count=2,
    )
    strong_second = _make_spec(
        sv_dcid="Count_Person_Employed",
        purpose_dcid="DAC/Education",
        place_dcid="country/ETH",
        member_count=1,
    )

    part = build_candidates_part(
        [weak, strong_first, strong_second],
        variable_text="test variable",
        warnings=[],
        max_candidates=2,
        candidate_scores={weak.spec_id: 0.7},
    )

    assert isinstance(part, CandidatesPart)
    assert [spec.spec_id for spec in part.candidates.specs] == [
        strong_first.spec_id,
        strong_second.spec_id,
    ]
    assert part.candidates.truncated is True


def test_assemble_candidates_ordering_is_intent_rank_when_bind_ranked():
    """CandidateSet.ordering is "intent_rank" on the bind_ranked=True path,
    the second of the two v1 ordering values: bind's own
    elected-then-also_plausible order."""
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
        bind_ranked=True,
    )

    assert isinstance(part, CandidatesPart)
    assert part.candidates.ordering == "intent_rank"


# Every input permutation must produce the same sort order.


def test_assemble_candidates_broadest_first_deterministic_under_shuffle():
    """Tied member counts use spec_id as a deterministic tie-break."""
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
        member_count=3,  # Tie with spec_b exercises the spec_id tie-break.
    )
    specs = [spec_a, spec_b, spec_c]
    expected_ids = [
        s.spec_id for s in sorted(specs, key=lambda s: (-s.resolved.shape.member_count, s.spec_id))
    ]

    for perm in itertools.permutations(specs):
        part = build_candidates_part(
            list(perm),
            variable_text="test variable",
            warnings=[],
        )
        assert isinstance(part, CandidatesPart)
        assert [s.spec_id for s in part.candidates.specs] == expected_ids


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
        "ETH's ODA Grants for malariacontrol. ETH's ODA Grants for education."
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


# Compose NEAR_NEIGHBOR_AMBIGUOUS from labels shown in the final candidate set.


def _relabel(spec, label: str):
    """Override a built Spec's own wire shape label for a test double."""
    relabeled_shape = spec.resolved.shape.model_copy(update={"label": label})
    return spec.model_copy(
        update={"resolved": spec.resolved.model_copy(update={"shape": relabeled_shape})}
    )


def test_near_neighbor_message_quotes_shown_labels_not_internal_metrics():
    """The warning reports shown labels and omits internal scores."""
    spec_chosen = _relabel(
        _make_spec(sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH"),
        "Forest Area",
    )
    spec_neighbor = _relabel(
        _make_spec(
            sv_dcid="Count_Household", purpose_dcid="DAC/Malariacontrol", place_dcid="country/ETH"
        ),
        "Count of Trees",
    )

    part = build_candidates_part(
        [spec_chosen, spec_neighbor],
        variable_text="test variable",
        warnings=[
            Warning(code=NEAR_NEIGHBOR_AMBIGUOUS, severity="info", message="placeholder"),
        ],
        near_neighbor_spec_ids=frozenset({spec_chosen.spec_id, spec_neighbor.spec_id}),
    )

    assert isinstance(part, CandidatesPart)
    warning = next(w for w in part.warnings if w.code == NEAR_NEIGHBOR_AMBIGUOUS)
    assert "Forest Area" in warning.message
    assert "Count of Trees" in warning.message
    assert "placeholder" not in warning.message
    # No cosine, margin or similarity number leaks into the reader-facing text.
    for _token in (".", "0."):
        assert not any(c.isdigit() for c in warning.message), warning.message


def test_near_neighbor_warning_dropped_when_named_shape_not_shown():
    """Drop the warning when fewer than two named shapes are returned."""
    spec_chosen = _make_spec(
        sv_dcid="Count_Person", purpose_dcid="DAC/Health", place_dcid="country/ETH"
    )
    spec_other = _make_spec(
        sv_dcid="Count_Household", purpose_dcid="DAC/Malariacontrol", place_dcid="country/ETH"
    )

    part = build_candidates_part(
        [spec_chosen, spec_other],
        variable_text="test variable",
        warnings=[
            Warning(code=NEAR_NEIGHBOR_AMBIGUOUS, severity="info", message="placeholder"),
        ],
        # Only spec_chosen is one of the two shapes the gate actually named.
        # The named neighbor is absent from this set, for example when it lacks data.
        near_neighbor_spec_ids=frozenset({spec_chosen.spec_id, "spec_never_shown"}),
    )

    assert isinstance(part, CandidatesPart)
    assert not any(w.code == NEAR_NEIGHBOR_AMBIGUOUS for w in part.warnings)
