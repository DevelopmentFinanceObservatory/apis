"""Platform-wide environment settings: CORS, shared executor capacity and the public website."""

from __future__ import annotations

from urllib.parse import urlsplit

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def env_validation_error_message(exc: ValidationError, *, env_prefix: str) -> str:
    """Format each validation error with its environment variable and Pydantic's reason."""
    lines = [
        f"invalid env var {env_prefix}{'.'.join(str(part) for part in error['loc']).upper()}: "
        f"{error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(lines)


class _PlatformSettings(BaseSettings):
    """Read at call time, not at import: in the deployed factory
    (dfo_apis.app), this module is imported before search.engine.config,
    whose own import triggers the repo-root .env seeding. Constructing
    this at import time would run before that seeding and silently ignore
    a value set only in .env. `parse_cors_origins`/`executor_max_workers`
    each construct a fresh instance rather than caching one at import.
    """

    model_config = SettingsConfigDict(env_prefix="DFO_")

    cors_origins: str = ""
    executor_max_workers: int = Field(default=32, ge=1)


def parse_cors_origins() -> list[str]:
    """Parse comma-separated `DFO_CORS_ORIGINS`. An empty value adds no CORS headers.

    Read at app creation time so environment changes after import are honoured."""
    raw = _PlatformSettings().cors_origins
    return [o.strip() for o in raw.split(",") if o.strip()]


def executor_max_workers() -> int:
    """Shared executor capacity (see dfo_platform.executor / dfo_platform.app)."""
    return _PlatformSettings().executor_max_workers


class PublicSiteSettings(BaseSettings):
    """The public website both surfaces link to from download bundles.

    `public_site_url` (`DFO_PUBLIC_SITE_URL`, required) is the website origin, which also
    proxies `/api/*`. Bundles print every link under it. `site_dataset_links`
    (`DFO_SITE_DATASET_LINKS`, default false) adds the website's `/data?spec=&v=` page to
    bundles of a standalone definition with a described identity, once the website serves
    that route. A pinned definition (`kind: "stat_var"`) gets none, because the website
    replays the identity through Search, which refuses the pinned arm.
    """

    model_config = SettingsConfigDict(env_prefix="DFO_")

    public_site_url: str = Field(min_length=1)
    site_dataset_links: bool = False

    @field_validator("public_site_url")
    @classmethod
    def _origin(cls, value: str) -> str:
        """An `http` or `https` origin with the trailing slash removed. User info, a path, query
        or fragment is refused, because bundles append their own paths and print the origin."""
        parts = urlsplit(value.rstrip("/"))
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be an http or https origin, such as https://example.org")
        if parts.username is not None or parts.password is not None:
            raise ValueError("must be an origin without user info")
        if parts.path or parts.query or parts.fragment:
            raise ValueError("must be an origin without a path, query or fragment")
        try:
            port = parts.port
        except ValueError:
            raise ValueError("must be an origin with a port from 0 to 65535") from None
        host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
        return f"{parts.scheme}://{host}" + (f":{port}" if port is not None else "")

    @classmethod
    def from_env(cls) -> PublicSiteSettings:
        """Read both variables. Raises `ValueError` naming the variable at fault."""
        try:
            return cls()
        except ValidationError as exc:
            raise ValueError(env_validation_error_message(exc, env_prefix="DFO_")) from exc
