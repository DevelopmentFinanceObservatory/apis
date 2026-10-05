"""The key under which `llm_responses.json` stores one recorded LLM response."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


def schema_fingerprint(schema: type[BaseModel]) -> str:
    """First 12 hex digits of the sha1 of `schema`'s canonical JSON Schema."""
    canonical = json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def fixture_key(schema: type[BaseModel], system: str, prompt: str) -> str:
    """`Name@fingerprint:sha1(system + chr(1) + prompt)`.

    The fingerprint moves whenever the schema's fields change, so a recorded response can
    never replay into a schema it was not produced under.
    """
    digest = hashlib.sha1((system + "\x01" + prompt).encode()).hexdigest()
    return f"{schema.__name__}@{schema_fingerprint(schema)}:{digest}"
