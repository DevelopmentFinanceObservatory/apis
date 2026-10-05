"""Connect to the `presentation` database with the IAM runtime role or password migrator
role, using Cloud SQL Connector or TCP. Settings reuse `dataset_store.connection` under the
`PRESENTATION_STORE_` prefix. Transport failures retain `DatasetStoreConnectionError`.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from dataset_store.connection import BaseConnectionSettings
from presentation_store.errors import PresentationStoreConfigError

__all__ = ["PresentationStoreSettings"]


class PresentationStoreSettings(BaseConnectionSettings):
    """`BaseConnectionSettings` under the `PRESENTATION_STORE_` prefix.
    Mirrors `dataset_store.connection.ConnectionSettings` field-for-field,
    with its own `database` default and a smaller `pool_size` default.
    """

    model_config = SettingsConfigDict(env_prefix="PRESENTATION_STORE_", frozen=True)

    config_error: ClassVar[type[Exception]] = PresentationStoreConfigError

    database: str = "presentation"
    pool_size: int = Field(default=2, ge=1)
