"""Own the process-wide, reference-counted presentation database pool, separate from
`dataset_store.provider`'s pool.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from dataset_store.connection import build_connector
from dataset_store.pool import ConnectionPool
from presentation_store.connection import PresentationStoreSettings
from presentation_store.store_errors import is_broken

if TYPE_CHECKING:
    from google.cloud.sql.connector import Connector

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pool: ConnectionPool | None = None
_connector: Connector | None = None
_borrowers = 0


def shared_pool() -> ConnectionPool:
    """Return the shared pool. Use the Cloud SQL Connector when its instance name is set,
    otherwise plain TCP. The pool discards a connection whose failure the store translated
    into `PresentationStoreUnavailable` (`store_errors.is_broken`). Register each borrow after
    construction succeeds.
    """
    global _pool, _connector, _borrowers
    with _lock:
        if _pool is None:
            settings = PresentationStoreSettings.from_env(
                user_var="PRESENTATION_STORE_RUNTIME_DB_USER",
                optional_password_var="PRESENTATION_STORE_RUNTIME_DB_PASSWORD",
                enable_iam_auth=True,
            )
            _connector = build_connector() if settings.instance_connection_name else None
            _pool = ConnectionPool(
                connect=settings.connect_callable(connector=_connector),
                size=settings.pool_size,
                is_broken=is_broken,
            )
        _borrowers += 1
        return _pool


def close_shared_pool() -> None:
    """Return one pool borrow. Close the pool and connector after the last borrow. Extra calls
    log a warning without raising.
    """
    global _pool, _connector, _borrowers
    with _lock:
        if _borrowers > 0:
            _borrowers -= 1
        else:
            logger.warning(
                "close_shared_pool() called with zero outstanding borrowers -- some "
                "caller's shared_pool()/close_shared_pool() accounting is unbalanced"
            )
        if _borrowers > 0:
            return
        if _pool is not None:
            _pool.close_all()
            _pool = None
        if _connector is not None:
            _connector.close()
            _connector = None
