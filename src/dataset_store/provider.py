"""One process-wide pool for every dataset-store consumer.

Calls to `shared_pool()` register borrows. The pool and Connector close
when matching `close_shared_pool()` calls release them all. Independent
FastAPI lifespans can therefore share the pool safely.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from dataset_store.connection import ConnectionSettings, build_connector
from dataset_store.pool import ConnectionPool

if TYPE_CHECKING:
    from google.cloud.sql.connector import Connector

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_pool: ConnectionPool | None = None
_connector: Connector | None = None
_borrowers = 0


def shared_pool() -> ConnectionPool:
    """Return the process pool, building it from runtime settings on first use.

    Each successful call registers one borrow. Failed construction registers none.
    """
    global _pool, _connector, _borrowers
    with _lock:
        if _pool is None:
            settings = ConnectionSettings.from_env(
                user_var="DATASET_STORE_RUNTIME_DB_USER",
                optional_password_var="DATASET_STORE_RUNTIME_DB_PASSWORD",
                enable_iam_auth=True,
            )
            _connector = build_connector() if settings.instance_connection_name else None
            _pool = ConnectionPool(
                connect=settings.connect_callable(connector=_connector),
                size=settings.pool_size,
                connect_timeout=settings.connect_timeout,
            )
        _borrowers += 1
        return _pool


def close_shared_pool() -> None:
    """Release one borrow. Close idle connections and Connector at zero.

    The final release clears shared state for the next `shared_pool()` call.
    Connections in use remain with their callers. A release without an
    outstanding borrow logs a warning.
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
