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
                    "entities": [{"text": "USA", "role_text": "from", "dcid": None}],
                    "time_text": "2015 to 2024",
                    "unit_text": "constant 2023 USD",
                    "source_text": "OECD CRS",
                    "operations": [
                        {
                            "verb": "rank",
                            "phrase": "top 5 donors",
                            "over": "origin",
                            "target_year": None,
                        }
                    ],
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

    def test_empty_dcid_refuses(self):
        with pytest.raises(ValidationError):
            ParsedEntity.model_validate({"text": "Congo", "dcid": ""})

    def test_unknown_nested_field_refuses_through_part(self):
        with pytest.raises(ValidationError):
            ParsedPart.model_validate(
                {"variable_text": "GDP", "entities": [{"text": "USA", "unexpected": "x"}]}
            )
