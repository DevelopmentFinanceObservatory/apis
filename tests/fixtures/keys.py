"""The key under which `llm_responses.json` stores one recorded LLM response."""

from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel


def _contract(schema: type[BaseModel]) -> type[BaseModel]:
    """The root model of `schema`'s name in its MRO.

    `bind_schema(size)` subclasses `_UniversalBindOutput` under the same name to cap
    `measuring` at the menu's size. The menu is in the prompt the key hashes, so every size
    shares one reply contract and one fingerprint.
    """
    return next(
        base
        for base in reversed(schema.__mro__)
        if issubclass(base, BaseModel) and base.__name__ == schema.__name__
    )


def schema_fingerprint(schema: type[BaseModel]) -> str:
    """First 12 hex digits of the sha1 of the canonical JSON Schema of `schema`'s contract."""
    canonical = json.dumps(
        _contract(schema).model_json_schema(), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def fixture_key(schema: type[BaseModel], system: str, prompt: str) -> str:
    """`Name@fingerprint:sha1(system + chr(1) + prompt)`.

    The fingerprint moves whenever the contract's fields change, so a recorded response can
    never replay into a schema it was not produced under.
    """
    digest = hashlib.sha1((system + "\x01" + prompt).encode()).hexdigest()
    return f"{schema.__name__}@{schema_fingerprint(schema)}:{digest}"
