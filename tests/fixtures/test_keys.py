"""Fixture keys carry the response schema's fingerprint."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from search.engine.bind import _UniversalBindOutput
from search.engine.dataset_mention import MentionReadings
from search.engine.extract import Extraction
from tests.fixtures.keys import fixture_key


class _Answer(BaseModel):
    value: str


def _answer_with_extra_field() -> type[BaseModel]:
    class _Answer(BaseModel):
        value: str
        confidence: float = 0.0

    return _Answer


def test_a_schema_field_change_moves_the_key_and_keeps_the_prompt_digest():
    before = fixture_key(_Answer, "system", "prompt")
    after = fixture_key(_answer_with_extra_field(), "system", "prompt")

    assert before != after
    assert before.split("@")[0] == after.split("@")[0] == "_Answer"
    assert before.split(":")[1] == after.split(":")[1]


def test_a_prompt_change_moves_the_key_under_one_schema():
    assert fixture_key(_Answer, "system", "a") != fixture_key(_Answer, "system", "b")
    assert fixture_key(_Answer, "system", "a") == fixture_key(_Answer, "system", "a")


def test_every_recorded_key_is_a_current_fixture_key_shape():
    bank = json.loads((Path(__file__).parent / "llm_responses.json").read_text())
    pattern = re.compile(
        r"^(Extraction|MentionReadings|_UniversalBindOutput)@[0-9a-f]{12}:[0-9a-f]{40}$"
    )
    keys = [key for key in bank if not key.startswith("_comment")]
    assert keys
    assert [key for key in keys if not pattern.match(key)] == []
    current = {
        fixture_key(schema, "", "").split(":")[0]
        for schema in (Extraction, MentionReadings, _UniversalBindOutput)
    }
    assert {key.split(":")[0] for key in keys} == current
