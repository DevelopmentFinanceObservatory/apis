"""Test definite-part sentence rendering."""

from __future__ import annotations

from search.engine.assemble import build_definite_part
from search.models import Headline, HeadlineYear
from search.render.plan import render_sentence_plan
from search.render.tokens import sentence_from_plan
from tests.engine._assemble_fixtures import _make_spec


def test_build_definite_part_renders_sentence_when_requested():
    spec_health = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )

    part = build_definite_part(
        spec_health,
        variable_text="test variable",
        warnings=[],
        include_sentence=True,
    )

    assert part.spec.resolved.rendered_sentence is not None
    assert part.spec.resolved.rendered_sentence == sentence_from_plan(
        part.spec.resolved.sentence_plan
    )


def test_build_definite_part_no_sentence_when_not_requested():
    spec_health = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )

    part = build_definite_part(
        spec_health,
        variable_text="test variable",
        warnings=[],
    )

    assert part.spec.resolved.rendered_sentence is None
    assert part.rendered_sentence is None
    assert part.sentence_plan is None


def test_a_definite_spec_sentence_is_its_heading_and_byline_alone():
    """The departures belong to the part, so the spec's own plan is the rendered plan with
    no note after it. The coverage headline stays on `resolved.headline`."""
    spec_health = _make_spec(
        sv_dcid="Count_Person",
        purpose_dcid="DAC/Health",
        place_dcid="country/ETH",
    )
    headline = Headline(
        year=2024,
        year_named=False,
        observations=3,
        series=3,
        role_counts={},
        recent_years=[HeadlineYear(year=2024, observations=3, series=3, role_counts={})],
    )
    spec_health = spec_health.model_copy(
        update={"resolved": spec_health.resolved.model_copy(update={"headline": headline})}
    )

    part = build_definite_part(
        spec_health,
        variable_text="test variable",
        warnings=[],
        include_sentence=True,
    )

    assert part.spec.resolved.sentence_plan == render_sentence_plan(spec_health)
    assert "observation" not in (part.spec.resolved.rendered_sentence or "")
