"""Build identity for the `x-dfo-commit` response header.

The image sets `DFO_BUILD_COMMIT` from the deploy workflow's `github.sha`. Containers have
no git checkout. Local runs without the variable use `"unknown"`.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class _BuildSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DFO_")

    build_commit: str = ""


def build_commit() -> str:
    """Read `DFO_BUILD_COMMIT` at call time. Return `"unknown"` when unset or empty."""
    return _BuildSettings().build_commit or "unknown"
