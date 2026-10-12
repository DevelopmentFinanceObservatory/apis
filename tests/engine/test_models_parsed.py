"""Model validation for the parsed door's request contract: ParsedInput,
ParsedPart, ParsedEntity. Pure pydantic tests -- no engine, no I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from search.engine.config import (
    SEARCH_MAX_ENTITY_TEXT_CHARS,
    SEARCH_MAX_PART_ENTITIES,
    SEARCH_MAX_PARTS,
    SEARCH_MAX_ROLE_TEXT_CHARS,
    SEARCH_MAX_SOURCE_TEXT_CHARS,
    SEARCH_MAX_TIME_TEXT_CHARS,
    SEARCH_MAX_UNIT_TEXT_CHARS,
    SEARCH_MAX_VARIABLE_TEXT_CHARS,
    settings,
)
from search.models import ParsedEntity, ParsedInput, ParsedPart


def _min_part(**overrides) -> dict:
    part = {"variable_text": "health ODA grants"}
    part.update(overrides)
    return part


class TestCapParity:
    """search.models can't import search.engine.config (search.engine's package
    init pulls in the engine core, which would make base model usage require
    the engine extra), so ParsedPart/ParsedEntity/ParsedInput bake their own
    literal caps. These assertions are the canary against the two sets drifting.
    """

    @pytest.mark.parametrize(
        ("field", "metadata_index", "config_limit"),
        [
            pytest.param(
                ParsedInput.model_fields["parts"],
                1,
                SEARCH_MAX_PARTS,
                id="max_parts_matches_config",
            ),
            pytest.param(
                ParsedPart.model_fields["entities"],
                0,
                SEARCH_MAX_PART_ENTITIES,
                id="max_part_entities_matches_config",
            ),
            pytest.param(
                ParsedPart.model_fields["variable_text"],
                1,
                SEARCH_MAX_VARIABLE_TEXT_CHARS,
                id="max_variable_text_chars_matches_config",
            ),
            pytest.param(
                ParsedEntity.model_fields["text"],
                1,
                SEARCH_MAX_ENTITY_TEXT_CHARS,
                id="max_entity_text_chars_matches_config",
            ),
            pytest.param(
                ParsedEntity.model_fields["role_text"],
                1,
                SEARCH_MAX_ROLE_TEXT_CHARS,
                id="max_role_text_chars_matches_config",
            ),
            pytest.param(
                ParsedPart.model_fields["time_text"],
                1,
                SEARCH_MAX_TIME_TEXT_CHARS,
                id="max_time_text_chars_matches_config",
            ),
            pytest.param(
                ParsedPart.model_fields["unit_text"],
                1,
                SEARCH_MAX_UNIT_TEXT_CHARS,
                id="max_unit_text_chars_matches_config",
            ),
            pytest.param(
                ParsedPart.model_fields["source_text"],
                1,
                SEARCH_MAX_SOURCE_TEXT_CHARS,
                id="max_source_text_chars_matches_config",
            ),
            pytest.param(
                ParsedInput.model_fields["raw_query"],
                1,
                settings().max_query_chars,
                id="max_raw_query_chars_matches_config",
            ),
        ],
    )
    def test_field_caps_match_config(self, field, metadata_index, config_limit):
        assert field.metadata[metadata_index].max_length == config_limit


class TestParsedInputParts:
    def test_empty_parts_refuses(self):
        with pytest.raises(ValidationError):
            ParsedInput.model_validate({"parts": []})

    def test_parts_over_max_refuses(self):
        with pytest.raises(ValidationError):
            ParsedInput.model_validate({"parts": [_min_part()] * (SEARCH_MAX_PARTS + 1)})

    def test_parts_at_max_accepts(self):
        req = ParsedInput.model_validate({"parts": [_min_part()] * SEARCH_MAX_PARTS})
        assert len(req.parts) == SEARCH_MAX_PARTS

    def test_unknown_top_level_field_refuses(self):
        with pytest.raises(ValidationError):
            ParsedInput.model_validate({"parts": [_min_part()], "unexpected": "x"})

    def test_empty_raw_query_refuses(self):
        with pytest.raises(ValidationError):
            ParsedInput.model_validate({"parts": [_min_part()], "raw_query": ""})

    def test_oversized_raw_query_refuses(self):
        with pytest.raises(ValidationError):
            ParsedInput.model_validate(
                {"parts": [_min_part()], "raw_query": "x" * (settings().max_query_chars + 1)}
            )

    def test_minimal_well_formed_round_trips(self):
        payload = {
            "kind": "parsed",
            "parts": [
                {
                    "variable_text": "health ODA grants",
                    "entities": [
                        {"text": "USA", "role_text": "from", "dcid": None, "members": None}
                    ],
                    "time_text": "2015 to 2024",
                    "unit_text": "constant 2023 USD",
                    "sources": [
                        {"text": "IDS", "ref": None, "polarity": "include"},
                        {"text": "CRS", "ref": None, "polarity": "exclude"},
                        {"text": None, "ref": "provenance/SomeTable", "polarity": "include"},
                        {"text": "OECD CRS", "ref": None, "polarity": "include"},
                    ],
                    "source_text": "OECD CRS",
                    "provenance": "provenance/SomeTable",
                    "operations": [
                        {
                            "verb": "rank",
                            "phrase": "top 5 donors",
                            "over": "origin",
                            "target_year": None,
                        }
                    ],
                    "open_roles": ["recipient"],
                }
            ],
            "raw_query": "health ODA grants from USA since 2015",
        }
        req = ParsedInput.model_validate(payload)
        assert req.model_dump(mode="json") == payload


class TestParsedPartFields:
    def test_empty_variable_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": ""})

    def test_oversized_variable_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": "x" * (SEARCH_MAX_VARIABLE_TEXT_CHARS + 1)})

    def test_empty_time_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": "GDP", "time_text": ""})

    def test_oversized_time_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {"variable_text": "GDP", "time_text": "x" * (SEARCH_MAX_TIME_TEXT_CHARS + 1)}
            )

    def test_empty_unit_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": "GDP", "unit_text": ""})

    def test_oversized_unit_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {"variable_text": "GDP", "unit_text": "x" * (SEARCH_MAX_UNIT_TEXT_CHARS + 1)}
            )

    def test_empty_source_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": "GDP", "source_text": ""})

    def test_oversized_source_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {"variable_text": "GDP", "source_text": "x" * (SEARCH_MAX_SOURCE_TEXT_CHARS + 1)}
            )

    @pytest.mark.parametrize(
        "mention",
        [{}, {"polarity": "exclude"}, {"text": ""}, {"text": "x" * 121}, {"ref": ""}],
        ids=["empty", "polarity only", "blank text", "long text", "blank ref"],
    )
    def test_a_mention_without_text_or_ref_refuses(self, mention: dict):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": "GDP", "sources": [mention]})

    @pytest.mark.parametrize(
        "deprecated",
        [{}, {"provenance": "provenance/X"}, {"source_text": "OECD"}],
        ids=["sources", "sources-and-provenance", "sources-and-source-text"],
    )
    def test_more_than_eight_mentions_refuse_counting_the_deprecated_fields(self, deprecated: dict):
        posted = 9 if not deprecated else 8
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {
                    "variable_text": "GDP",
                    "sources": [{"text": f"IDS {i}"} for i in range(posted)],
                    **deprecated,
                }
            )

    def test_the_deprecated_fields_fold_into_sources_after_the_posted_mentions(self):
        part = ParsedPart.model_validate(
            {
                "variable_text": "GDP",
                "sources": [{"text": "CRS", "polarity": "exclude"}],
                "provenance": "provenance/X",
                "source_text": "OECD",
            }
        )
        assert [(m.text, m.ref, m.polarity) for m in part.sources] == [
            ("CRS", None, "exclude"),
            (None, "provenance/X", "include"),
            ("OECD", None, "include"),
        ]
        assert [part.mention_field(i) for i in range(3)] == [
            "sources/0",
            "provenance",
            "source_text",
        ]

    def test_a_dump_posted_back_folds_to_the_same_sources(self):
        part = ParsedPart(variable_text="GDP", provenance="provenance/X", source_text="OECD")
        again = ParsedPart.model_validate(part.model_dump(mode="json"))
        assert again.sources == part.sources
        assert len(again.sources) == 2

    def test_entities_over_max_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {
                    "variable_text": "GDP",
                    "entities": [{"text": "x"}] * (SEARCH_MAX_PART_ENTITIES + 1),
                }
            )

    def test_entities_at_max_accepts(self):
        part = ParsedPart.model_validate(
            {"variable_text": "GDP", "entities": [{"text": "x"}] * SEARCH_MAX_PART_ENTITIES}
        )
        assert part.entities is not None
        assert len(part.entities) == SEARCH_MAX_PART_ENTITIES

    def test_unknown_field_refuses(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate({"variable_text": "GDP", "unexpected": "x"})


class TestParsedEntityFields:
    def test_empty_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate({"text": ""})

    def test_oversized_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate({"text": "x" * (SEARCH_MAX_ENTITY_TEXT_CHARS + 1)})

    def test_empty_role_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate({"text": "USA", "role_text": ""})

    def test_oversized_role_text_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate(
                {"text": "USA", "role_text": "x" * (SEARCH_MAX_ROLE_TEXT_CHARS + 1)}
            )

    def test_unknown_field_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate({"text": "USA", "unexpected": "x"})

    def test_dcid_defaults_to_none(self):
        entity = ParsedEntity.model_validate({"text": "Congo"})
        assert entity.dcid is None

    @pytest.mark.parametrize("posted", [{}, {"members": False}, {"members": None}])
    def test_an_entity_that_does_not_ask_for_members_carries_no_members(self, posted):
        """Omitted, false and null are one unset state, and the wire dump, which drops
        nulls, omits it."""
        entity = ParsedEntity.model_validate({"text": "Africa", **posted})
        assert entity == ParsedEntity(text="Africa")
        assert entity.members is None
        assert entity.model_dump(mode="json", exclude_none=True) == {"text": "Africa"}

    def test_an_entity_that_asks_for_members_states_it(self):
        entity = ParsedEntity.model_validate({"text": "Africa", "members": True})
        assert entity.model_dump(mode="json", exclude_none=True) == {
            "text": "Africa",
            "members": True,
        }
        assert ParsedEntity.model_validate(entity.model_dump(mode="json")) == entity

    def test_empty_dcid_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate({"text": "Congo", "dcid": ""})

    def test_unknown_nested_field_refuses_through_part(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {"variable_text": "GDP", "entities": [{"text": "USA", "unexpected": "x"}]}
            )
