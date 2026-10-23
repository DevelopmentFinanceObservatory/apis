"""Tests fan-out restatement of direction and absence reasons (#625).

Directional legs echo each entity's role. A name bound on both axes has no restatement because
`ParsedEntity` stores one role. Tests use offline resolutions or `_leg_restatement` directly.
"""

from __future__ import annotations

from search.engine.core import _quick_no_data
from search.engine.envelope import now_ms
from search.engine.extract import _EXTRACTION_SYSTEM_PROMPT, Extraction, ExtractionResult
from search.engine.fanout import _leg_restatement, _placeholder_no_data
from search.engine.role_text import _ROLE_TEXT_TO_AXIS, DoorRole, role_axis_for_text
from search.models import (
    DatasetResubmitInput,
    ParsedInput,
    ParsedPart,
    ResolveRequest,
    ResubmitPlaceholderEntry,
)
from tests.engine._harness import make_request, offline_resolve
from tests.fixtures import FakeLLM, _load_json
from tests.fixtures.keys import fixture_key


def _extract_key(query: str) -> str:
    return fixture_key(Extraction, _EXTRACTION_SYSTEM_PROMPT, query)


def _llm_with_extraction(query: str, *, variables: list[str], entities: list[dict]) -> FakeLLM:
    """Add one query extraction with explicit entity role and span fields."""
    return FakeLLM(
        responses={
            **_load_json("llm_responses.json"),
            _extract_key(query): {
                "kind": "resolution",
                "variables": variables,
                "entities": entities,
                "dates": [],
            },
        }
    )


# The corpus's own directional two-entity query, whose extraction is already
# recorded, so no new fixture is minted: both ends are named, the provider on
# the origin axis and Kenya on the destination one.
_TWO_ENDED_QUERY = "ODA commitments from the Asian Development Bank to Kenya"


# ---------------------------------------------------------------------------
# A directional leg echoes the axis it resolved on
# ---------------------------------------------------------------------------


class TestDirectionalLegEchoesItsAxis:
    def test_raw_text_leg_echoes_a_recognised_role_text_per_entity(self):
        """The echoed role text maps back to the same axis on parsed input."""
        part = offline_resolve(make_request(_TWO_ENDED_QUERY)).parts[0]

        assert part.restatement is not None, (
            "a leg bound on directional axes must still echo its inputs"
        )
        echoed = {e.text: e.role_text for e in part.restatement.entities or []}
        assert echoed == {"Asian Development Bank": "from", "Kenya": "to"}
        for role_text in echoed.values():
            assert role_text in _ROLE_TEXT_TO_AXIS
        assert role_axis_for_text(echoed["Asian Development Bank"]) == "origin"
        assert role_axis_for_text(echoed["Kenya"]) == "destination"

    def test_the_echo_round_trips_through_the_parsed_door(self):
        """Posting a restatement back reproduces the same axes and status."""
        original = offline_resolve(make_request(_TWO_ENDED_QUERY)).parts[0]
        assert original.restatement is not None

        reparsed = offline_resolve(
            ResolveRequest(input=ParsedInput(parts=[original.restatement]))
        ).parts[0]
        assert reparsed.restatement == original.restatement
        assert reparsed.status == original.status

    def test_an_unspecified_entity_still_echoes_bare(self):
        """A leg whose whole entity list carries no axis is undirected, so its
        echo names no role_text and stays restatable."""
        part = offline_resolve(make_request("population and GDP in Brazil")).parts[0]
        assert part.restatement is not None
        assert [e.text for e in part.restatement.entities or []] == ["Brazil"]
        assert all(e.role_text is None for e in part.restatement.entities or [])


# ---------------------------------------------------------------------------
# The one-axis rule
# ---------------------------------------------------------------------------


class TestOneAxisRule:
    def test_a_name_on_both_axes_yields_no_echo(self):
        """One `role_text` cannot represent a self-flow's two axes."""
        restatement = _leg_restatement(
            "aid",
            entities=["Kenya"],
            date_request=None,
            unit_text=None,
            entity_roles={
                "Kenya": [
                    DoorRole(axis="origin", span="from Kenya"),
                    DoorRole(axis="destination", span="to Kenya"),
                ]
            },
        )
        assert restatement is None

    def test_a_name_on_one_axis_echoes_that_axis(self):
        restatement = _leg_restatement(
            "aid",
            entities=["Kenya"],
            date_request=None,
            unit_text=None,
            entity_roles={"Kenya": [DoorRole(axis="origin", span="from Kenya")]},
        )
        assert restatement == ParsedPart(
            variable_text="aid", entities=[{"text": "Kenya", "role_text": "from"}]
        )

    def test_two_names_on_different_axes_each_echo_their_own(self):
        restatement = _leg_restatement(
            "aid",
            entities=["France", "Kenya"],
            date_request=None,
            unit_text=None,
            entity_roles={
                "France": [DoorRole(axis="origin", span="France")],
                "Kenya": [DoorRole(axis="destination", span="Kenya")],
            },
        )
        assert restatement is not None
        assert [(e.text, e.role_text) for e in restatement.entities or []] == [
            ("France", "from"),
            ("Kenya", "to"),
        ]

    def test_a_placeholder_leg_echoes_the_axis_the_door_supplied(self):
        """A placeholder leg preserves the door's supplied axis."""
        region = _placeholder_no_data(
            "aid",
            0,
            reason="timeout",
            entities=["Kenya"],
            role_query="aid to Kenya",
            entity_roles={"Kenya": [DoorRole(axis="destination", span="Kenya")]},
        )
        assert region.restatement is not None
        assert [(e.text, e.role_text) for e in region.restatement.entities or []] == [
            ("Kenya", "to")
        ]


# ---------------------------------------------------------------------------
# Each case that has no faithful echo carries no restatement
# ---------------------------------------------------------------------------


class TestRestatementIsNullPerCase:
    def test_no_phrase_inputs_on_a_placeholder_entry(self):
        entry = ResubmitPlaceholderEntry(
            variable_text="unemployment rate", status="no_data", reason="no_observations"
        )
        part = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry]))).parts[0]
        assert part.restatement is None

    def test_no_phrase_inputs_on_an_identity_entry(self):
        """Identity entries contain no phrase inputs to echo."""
        from tests.engine.test_dataset_resubmit import _gdp_entry

        entry = _gdp_entry("country/IND", "GDP")
        part = offline_resolve(ResolveRequest(input=DatasetResubmitInput(entries=[entry]))).parts[0]
        assert part.status == "definite"
        assert part.restatement is None

    def test_direction_unnameable_for_a_leg_resolved_without_a_door_axis(self):
        """An unbound entity has no direction to echo."""
        query = "population in Nowherelandia"
        llm = _llm_with_extraction(
            query,
            variables=["population"],
            entities=[
                {"text": "Nowherelandia", "span": "Nowherelandia", "flow_role": "unspecified"}
            ],
        )
        part = offline_resolve(make_request(query), llm=llm).parts[0]
        assert part.restatement is None

    def test_inputs_exceed_caps_for_an_entity_over_the_parsed_doors_bound(self):
        """A 129-character entity exceeds the parsed door's 128-character cap."""
        long_entity = "N" * 129
        query = f"aid from {long_entity}"
        llm = _llm_with_extraction(
            query,
            variables=["aid"],
            entities=[
                {
                    "text": long_entity,
                    "span": long_entity,
                    "flow_role": "origin",
                    "role_span": "from",
                }
            ],
        )
        part = offline_resolve(make_request(query), llm=llm).parts[0]
        assert part.restatement is None

    def test_no_phrase_inputs_on_an_empty_query(self):
        """The whitespace-query shortcut has no phrase to echo."""
        part = offline_resolve(make_request("   ")).parts[0]
        assert part.restatement is None

    def test_no_phrase_inputs_on_no_variables_extracted(self, monkeypatch):
        """An empty extraction has no variable phrase to echo."""

        async def _empty_extraction(query: str, *, llm):
            return (
                ExtractionResult(
                    extraction=Extraction(kind="resolution", variables=[], entities=[], dates=[]),
                    qualifiers={},
                ),
                None,
            )

        monkeypatch.setattr("search.engine.entry.raw_text.extract", _empty_extraction)
        part = offline_resolve(make_request("asdf qwerty")).parts[0]
        assert part.restatement is None

    def test_no_phrase_inputs_on_the_quick_no_data_shortcut(self):
        """The unreachable quick no-data fallback has no phrase inputs to echo."""
        response = _quick_no_data(
            reason="variable_not_resolved",
            query="",
            engine_build="test",
            start_ms=now_ms(),
        )
        part = response.parts[0]
        assert part.restatement is None
