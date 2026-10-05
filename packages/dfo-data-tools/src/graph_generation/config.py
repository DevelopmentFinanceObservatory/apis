"""Graph-generation settings for cache lifetimes and token source.

Settings validate at import. Fields use their declared environment aliases. Invalid values raise
`ValueError` with the offending variable name."""

from __future__ import annotations

from typing import Self

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings

from dfo_platform.config import env_validation_error_message


class _GraphGenerationSettings(BaseSettings):
    # Cache lifetime for entries with a known generation token. At most one day, the lifetime of
    # the UN backend's generation (`origin_catalogue.graph.UnGeneration`), because caches keyed
    # on our generation also hold answers built from UN reads.
    cache_max_age_seconds: int = Field(
        default=86_400, ge=1, le=86_400, alias="GRAPH_CACHE_MAX_AGE_SECONDS"
    )

    # Shorter lifetime for unknown generations, which provide no invalidation signal.
    cache_unknown_generation_max_age_seconds: int = Field(
        default=3_600, ge=1, alias="GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS"
    )

    # Cloud Run Admin API v2 resource `projects/<p>/locations/<r>/services/<s>`.
    # The Admin API validates the resource shape. Malformed names use the unknown-generation path.
    generation_service: str = Field(default="", alias="DFO_GRAPH_GENERATION_SERVICE")

    # The fixed token skips HTTP and takes precedence over the Cloud Run service.
    generation: str = Field(default="", alias="DFO_GRAPH_GENERATION")

    @model_validator(mode="after")
    def _check_lifetimes(self) -> Self:
        # Unknown generations must expire no later than known generations.
        if self.cache_unknown_generation_max_age_seconds > self.cache_max_age_seconds:
            raise ValueError(
                "invalid env vars: GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS "
                f"({self.cache_unknown_generation_max_age_seconds}) must be <= "
                f"GRAPH_CACHE_MAX_AGE_SECONDS ({self.cache_max_age_seconds})"
            )
        return self


try:
    _settings = _GraphGenerationSettings()
except ValidationError as exc:
    raise ValueError(env_validation_error_message(exc, env_prefix="")) from exc

GRAPH_CACHE_MAX_AGE_SECONDS: int = _settings.cache_max_age_seconds
GRAPH_CACHE_UNKNOWN_GENERATION_MAX_AGE_SECONDS: int = (
    _settings.cache_unknown_generation_max_age_seconds
)
DFO_GRAPH_GENERATION_SERVICE: str = _settings.generation_service
DFO_GRAPH_GENERATION: str = _settings.generation
