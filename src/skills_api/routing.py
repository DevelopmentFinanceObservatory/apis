"""URL convertor and length ceiling for the Skills path parameters.

`dfo_skill_id` matches one path segment other than a name in `RESERVED_SKILL_IDS`, so a
literal route such as `/healthz` resolves regardless of router order. Any other segment
reaches the handler, which refuses one past `SKILL_ID_MAX_LENGTH` as `request_invalid` (400)
and answers an id outside the catalogue with `skill_not_found` (404). The resource path uses
Starlette's multi-segment `path` convertor and the index's `RESOURCE_PATH_MAX_LENGTH`.
"""

from __future__ import annotations

from starlette.convertors import Convertor, register_url_convertor

from skills_index.config import RESERVED_SKILL_IDS

# The longest `skill_id` the doors look up. A longer one refuses 400.
SKILL_ID_MAX_LENGTH = 512


class SkillIdConvertor(Convertor[str]):
    """Match one path segment, excluding reserved route names."""

    regex = r"(?!(?:" + "|".join(sorted(RESERVED_SKILL_IDS)) + r")(?:/|$))[^/]+"

    def convert(self, value: str) -> str:
        return value

    def to_string(self, value: str) -> str:
        return value


register_url_convertor("dfo_skill_id", SkillIdConvertor())
