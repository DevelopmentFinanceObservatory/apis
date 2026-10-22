"""Graph settings with safe defaults read at import and live-reader settings checked at startup.

Pool sizes and timeouts need no credentials, so tests can build an injected
surface offline. `GraphSettings.from_env` validates `GRAPH_DCP_BASE_URL`
when the lifespan creates a live reader. An empty URL fails startup. The mixer
serves SDMX under the same API root, so `GRAPH_DCP_BASE_URL` and
`GRAPH_DCP_TOKEN` also address the SDMX availability reads.
"""

from __future__ import annotations

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from dfo_platform.config import env_validation_error_message
from origin_catalogue.graph import UN_BASE_URL, UN_BASE_URL_ENV


class _GraphDefaults(BaseSettings):
    """Safe-default settings read and validated once at import."""

    model_config = SettingsConfigDict(env_prefix="GRAPH_")

    # Worker count of the DCP-read executor all DCP reads share. Reads past it are refused.
    node_pool_size: int = Field(default=4, ge=1)

    # Bounds the client's wait, queue time included. A running worker thread outlives it.
    resolve_timeout_seconds: int = Field(default=30, ge=1)

    # Bounds the client's wait for one `node_arcs_many` call.
    node_timeout_seconds: int = Field(default=10, ge=1)

    # Most source documents the documentation route keeps in its cache at once.
    provenance_doc_cache_size: int = Field(default=512, ge=1)

    # Most built node envelopes the node routes keep at once (`graph_api.node_cache`).
    node_entry_cache_size: int = Field(default=2048, ge=1)

    # HTTP timeout of the UN deployment's SDMX client the node reader uses.
    sdmx_timeout_seconds: int = Field(default=30, ge=1)


try:
    _defaults = _GraphDefaults()
except ValidationError as exc:
    raise ValueError(env_validation_error_message(exc, env_prefix="GRAPH_")) from exc

GRAPH_NODE_POOL_SIZE: int = _defaults.node_pool_size
GRAPH_RESOLVE_TIMEOUT_SECONDS: int = _defaults.resolve_timeout_seconds
GRAPH_NODE_TIMEOUT_SECONDS: int = _defaults.node_timeout_seconds
GRAPH_PROVENANCE_DOC_CACHE_SIZE: int = _defaults.provenance_doc_cache_size
GRAPH_NODE_ENTRY_CACHE_SIZE: int = _defaults.node_entry_cache_size
GRAPH_SDMX_TIMEOUT_SECONDS: int = _defaults.sdmx_timeout_seconds

# A DCP read that waits longer than this for a worker logs at WARNING (`log_queue_wait`).
DCP_QUEUE_WAIT_WARN_SECONDS: float = 1.0


class GraphSettings(BaseSettings):
    """Live-reader settings validated in the surface lifespan. An empty DCP URL fails there.
    `un_base_url` is the UN System Data Commons API root, which serves `undata/` dcids, read
    from `UN_GRAPH_BASE_URL` like every other surface's."""

    model_config = SettingsConfigDict(env_prefix="GRAPH_")

    dcp_base_url: str = Field(min_length=1)
    dcp_token: str = ""
    un_base_url: str = Field(default=UN_BASE_URL, min_length=1, validation_alias=UN_BASE_URL_ENV)

    @classmethod
    def from_env(cls) -> GraphSettings:
        try:
            return cls()
        except ValidationError as exc:
            raise ValueError(env_validation_error_message(exc, env_prefix="GRAPH_")) from exc
