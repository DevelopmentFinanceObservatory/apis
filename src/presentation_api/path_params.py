"""These checks reject malformed path IDs before store or GCS lookups. They enforce
character and length limits, so well-formed test IDs reach the store even when their shape
differs from minted IDs.

`ops_` spec IDs and `ocv_` content versions pass through. `FrameStore.load` classifies
unsupported pairs.
"""

from __future__ import annotations

import re

__all__ = [
    "PathParamInvalid",
    "require_component_id",
    "require_content_version",
    "require_spec_id",
]

_SPEC_ID = re.compile(r"^(spec|ops)_[a-z0-9]{1,64}$")
_CONTENT_VERSION = re.compile(r"^(cv|ocv)_[a-z0-9]{1,64}$")
_COMPONENT_ID = re.compile(r"^cmp_[A-Za-z0-9_-]{1,64}$")


class PathParamInvalid(Exception):
    """A path ID fails format validation and maps to non-retryable `request_invalid` (400)."""

    def __init__(self, *, name: str, value: str) -> None:
        self.name = name
        self.value = value
        super().__init__(f"{name} {value!r} is not a well-formed id")


def require_spec_id(spec_id: str) -> str:
    if not _SPEC_ID.fullmatch(spec_id):
        raise PathParamInvalid(name="spec_id", value=spec_id)
    return spec_id


def require_content_version(content_version: str) -> str:
    if not _CONTENT_VERSION.fullmatch(content_version):
        raise PathParamInvalid(name="content_version", value=content_version)
    return content_version


def require_component_id(component_id: str) -> str:
    if not _COMPONENT_ID.fullmatch(component_id):
        raise PathParamInvalid(name="component_id", value=component_id)
    return component_id
