"""Tests denominator fallback precedence against materialisation and no-data reasons."""

from __future__ import annotations

from search.engine.region.codes import _elect_no_data_reason


class TestElectNoDataReason:
    """Tests precedence among denominator fallback and live no-data reasons."""

    def test_earlier_candidate_beats_denominator_fallback(self):
        assert (
            _elect_no_data_reason("no_matching_slice", fallback="denominator_not_available")
            == "no_matching_slice"
        )

    def test_variable_not_resolved_beats_denominator_fallback(self):
        assert (
            _elect_no_data_reason("variable_not_resolved", fallback="denominator_not_available")
            == "variable_not_resolved"
        )

    def test_none_candidate_with_slice_pending_elects_no_matching_slice(self):
        """A pending slice deferral supplies ``no_matching_slice`` as the second candidate."""
        assert (
            _elect_no_data_reason(None, "no_matching_slice", fallback="denominator_not_available")
            == "no_matching_slice"
        )

    def test_no_live_candidates_falls_back_to_denominator(self):
        assert _elect_no_data_reason(None, None, fallback="denominator_not_available") == (
            "denominator_not_available"
        )

    def test_no_observations_candidate_loses_to_denominator_fallback(self):
        """The denominator fallback outranks ``no_observations``."""
        assert (
            _elect_no_data_reason("no_observations", fallback="denominator_not_available")
            == "denominator_not_available"
        )

    def test_no_matching_slice_beats_no_observations_fallback(self):
        assert (
            _elect_no_data_reason("no_matching_slice", fallback="no_observations")
            == "no_matching_slice"
        )

    def test_facet_not_available_loses_to_no_observations_fallback(self):
        """The ``no_observations`` fallback outranks ``facet_not_available``."""
        assert (
            _elect_no_data_reason("facet_not_available", fallback="no_observations")
            == "no_observations"
        )

    def test_grounded_near_match_drops_variable_not_resolved(self):
        """A grounded near-match removes ``variable_not_resolved``."""
        assert (
            _elect_no_data_reason(
                "variable_not_resolved",
                fallback="no_observations",
                grounded_near_match=True,
            )
            == "no_observations"
        )

    def test_without_grounded_near_match_variable_not_resolved_threads(self):
        """A false near-match flag preserves ``variable_not_resolved``."""
        assert (
            _elect_no_data_reason(
                "variable_not_resolved",
                fallback="no_observations",
                grounded_near_match=False,
            )
            == "variable_not_resolved"
        )

    def test_fallback_alone_is_never_filtered_by_coherence_rule(self):
        """A near-match preserves fallback eligibility.

        A result is always returned, even when the fallback is in the incoherent set.
        """
        assert (
            _elect_no_data_reason(fallback="no_observations", grounded_near_match=True)
            == "no_observations"
        )
