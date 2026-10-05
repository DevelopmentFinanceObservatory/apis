"""Pins per-part parsed-input restatements, including whitespace-only rejection."""

from __future__ import annotations

from search.engine.entry.parsed import _part_restatement
from search.models import Operation, ParsedEntity, ParsedInput, ParsedPart, ResolveRequest
from tests.engine._harness import offline_resolve


def _a_parsed_part(**overrides) -> ParsedPart:
    """A part with values in every field copied by `_part_restatement`."""
    fields = {
        "variable_text": "fertility rate",
        "entities": [ParsedEntity(text="Kenya", role_text="in")],
        "time_text": "2020",
        "unit_text": "US dollars",
        "source_text": "World Bank",
        "operations": [Operation(verb="rank", phrase="top 5", over="place")],
    }
    fields.update(overrides)
    return ParsedPart(**fields)


class TestTheOverrideEchoesTheLegsOwnInputs:
    def test_a_parsed_part_comes_back_equal_on_every_carried_field(self):
        """A stripped variable text leaves all other posted fields unchanged."""
        posted = _a_parsed_part()
        restatement = _part_restatement(posted, variable_text="fertility rate")

        assert restatement is not None
        assert restatement == posted
        assert restatement.variable_text == "fertility rate"
        assert [(e.text, e.role_text) for e in restatement.entities or []] == [("Kenya", "in")]
        assert restatement.time_text == "2020"
        assert restatement.unit_text == "US dollars"
        assert restatement.source_text == "World Bank"
        assert restatement.operations == [Operation(verb="rank", phrase="top 5", over="place")]

    def test_the_stripped_text_is_what_lands_in_the_echo(self):
        """The restatement echoes the stripped text passed to `resolve_variable`."""
        posted = _a_parsed_part(variable_text="fertility rate\n")
        restatement = _part_restatement(posted, variable_text="fertility rate")

        assert restatement is not None
        assert restatement.variable_text == "fertility rate"


class TestTheWhitespaceOnlyEdge:
    def test_an_all_whitespace_text_yields_no_restatement(self):
        """Stripping whitespace-only text yields no restatement."""
        posted = ParsedPart(variable_text="   ")

        assert _part_restatement(posted, variable_text="") is None

    def test_the_same_part_still_echoes_when_its_text_is_real(self):
        """A non-empty text yields a restatement."""
        restatement = _part_restatement(
            ParsedPart(variable_text="population"), variable_text="population"
        )
        assert restatement is not None
        assert restatement.variable_text == "population"


class TestThroughTheRealParsedDoor:
    def test_two_parts_each_echo_with_no_reason(self):
        """Each echoed part has its own restatement and no refusal reason."""
        parts = [
            ParsedPart(variable_text="population", entities=[ParsedEntity(text="Brazil")]),
            ParsedPart(variable_text="GDP", entities=[ParsedEntity(text="Brazil")]),
        ]
        result = offline_resolve(ResolveRequest(input=ParsedInput(parts=parts)))

        assert len(result.parts) == 2
        for part in result.parts:
            assert part.restatement is not None
            assert part.restatement_reason is None
        assert [part.restatement.variable_text for part in result.parts] == ["population", "GDP"]

    def test_one_refused_echo_one_present_echo(self):
        """Whitespace-only text refuses one part while its sibling is restated."""
        parts = [
            ParsedPart(variable_text="   "),
            ParsedPart(variable_text="population", entities=[ParsedEntity(text="Brazil")]),
        ]
        result = offline_resolve(ResolveRequest(input=ParsedInput(parts=parts)))

        assert len(result.parts) == 2
        refused, echoed = result.parts
        assert refused.restatement is None
        assert refused.restatement_reason is not None
        assert echoed.restatement is not None
        assert echoed.restatement_reason is None
