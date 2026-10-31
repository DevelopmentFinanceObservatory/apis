"""Open Postgres connections as the IAM runtime role or password migrator role.

Connections use the Cloud SQL Connector over PSC or plain TCP. Only
Environment variables are read lazily through `ConnectionSettings.from_env()`.
"""

from __future__ import annotations

import contextlib
import os
import threading
from collections.abc import Callable
from typing import ClassVar, Protocol, Self, cast

import pg8000
import pg8000.exceptions
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dataset_store.dbapi import DBAPIConnection
from dataset_store.errors import DatasetStoreConfigError, DatasetStoreConnectionError


class _Connector(Protocol):
    """Connector interface kept local to avoid importing its optional dependency at module load."""

    def connect(
        self, instance_connection_string: str, driver: str, **kwargs: object
    ) -> DBAPIConnection: ...


def default_is_broken(exc: BaseException) -> bool:
    """Treat transport, OS, and store-connection errors as broken.

    Other errors preserve the connection.
    """
    return isinstance(exc, (pg8000.exceptions.InterfaceError, OSError, DatasetStoreConnectionError))


def build_connector():
    """Build the process's PSC connector.

    Lazy token refresh avoids a stale token after Cloud Run freeze/thaw. PSC is the only
    reachable path to dfo-dev-platform-postgres. Construction starts a background thread, so
    build it once per process, never at import.
    """
    from google.cloud.sql.connector import Connector, IPTypes

    return Connector(refresh_strategy="lazy", ip_type=IPTypes.PSC)


def _connect_via_connector(
    connector: _Connector,
    instance_connection_name: str,
    *,
    timeout: float,
    kwargs: dict[str, object],
) -> DBAPIConnection:
    """Bound the connector dial, which has no socket timeout of its own.

    A timed-out dial continues on its daemon thread. Any connection returned
    later is closed.
    """
    cv = threading.Condition()
    ready = False
    abandoned = False
    result: DBAPIConnection | None = None
    error: BaseException | None = None

    def _dial() -> None:
        nonlocal result, error, ready
        try:
            conn = connector.connect(instance_connection_name, "pg8000", timeout=timeout, **kwargs)
        except Exception as exc:
            with cv:
                error = exc
                ready = True
                cv.notify_all()
            return
        with cv:
            hand_over = not abandoned
            if hand_over:
                result = conn
                ready = True
            cv.notify_all()
        if not hand_over:
            with contextlib.suppress(Exception):
                conn.close()

    threading.Thread(target=_dial, daemon=True).start()

    with cv:
        if not cv.wait_for(lambda: ready, timeout=timeout):
            abandoned = True
            raise DatasetStoreConnectionError(
                f"connector.connect to {instance_connection_name!r} did not "
                f"return within {timeout}s"
            )

    if error is not None:
        raise error
    assert result is not None
    return result


def _bound_connector_socket_io(conn: DBAPIConnection, *, timeout: float) -> None:
    """pg8000 ignores `timeout` on supplied sockets, so set private `_usock` directly.

    This API is test-pinned. Missing `_usock` raises a connection error to
    keep query I/O bounded.
    """
    usock = getattr(conn, "_usock", None)
    if usock is None or not hasattr(usock, "settimeout"):
        raise DatasetStoreConnectionError(
            "connector-path connection has no usable _usock -- pg8000's "
            "private socket attribute contract broke"
        )
    usock.settimeout(timeout)


def connect(
    *,
    connector: object | None,
    instance_connection_name: str | None,
    host: str,
    port: int,
    database: str,
    user: str,
    password: str | None = None,
    enable_iam_auth: bool = False,
    timeout: float = 30.0,
) -> DBAPIConnection:
    """Open one pg8000 connection via Cloud SQL Connector or plain TCP.

    The connector path uses IAM when enabled. Plain TCP requires a password.
    Transport and authentication failures become `DatasetStoreConnectionError`.
    """
    try:
        if connector is not None and instance_connection_name:
            kwargs: dict[str, object] = {
                "user": user,
                "db": database,
                "enable_iam_auth": enable_iam_auth,
            }
            if password is not None:
                kwargs["password"] = password
            conn = _connect_via_connector(
                cast(_Connector, connector),
                instance_connection_name,
                timeout=timeout,
                kwargs=kwargs,
            )
            _bound_connector_socket_io(conn, timeout=timeout)
            return conn
        if password is None:
            raise DatasetStoreConnectionError(
                "plain-TCP connect requires a password (no IAM fallback without the connector)"
            )
        return pg8000.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            timeout=timeout,
        )
    except DatasetStoreConnectionError:
        raise
    except Exception as exc:
        raise DatasetStoreConnectionError(
            f"failed to connect to {host}:{port}/{database} as {user!r}: {exc}"
        ) from exc


def _validation_error_message(exc: ValidationError, *, env_prefix: str) -> str:
    """Format validation errors with the prefixed environment-variable name."""
    lines = [
        f"invalid env var {env_prefix}{'.'.join(str(part) for part in error['loc']).upper()}: "
        f"{error['msg']}"
        for error in exc.errors()
    ]
    return "; ".join(lines)


class BaseConnectionSettings(BaseSettings):
    """Everything `connect()` needs, read once from the environment.
    Shared by `dataset_store.connection.ConnectionSettings` and
    `presentation_store.connection.PresentationStoreSettings`, which each
    set their own `env_prefix` (`DATASET_STORE_`/`PRESENTATION_STORE_`)
    and their own `database`/`pool_size` default. `user_var`/`password_var`
    (`from_env`'s own parameters) let each entrypoint point at its own
    role's env vars while sharing every other prefixed connection var.
    `config_error` names the exception type `from_env` raises for that
    subclass's prefix: `DatasetStoreConfigError` here,
    `PresentationStoreConfigError` for `PresentationStoreSettings`.
    """

    model_config = SettingsConfigDict(frozen=True)

    config_error: ClassVar[type[Exception]] = DatasetStoreConfigError

    instance_connection_name: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=5432, ge=1)
    database: str
    user: str = ""
    password: str | None = None
    enable_iam_auth: bool = False
    pool_size: int
    connect_timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)

    @field_validator("instance_connection_name", mode="before")
    @classmethod
    def _empty_string_is_unset(cls, value: object) -> object:
        return value or None

    @classmethod
    def from_env(
        cls,
        *,
        user_var: str,
        password_var: str | None = None,
        optional_password_var: str | None = None,
        enable_iam_auth: bool = False,
    ) -> Self:
        """Raise `config_error` with the environment variable and constraint for invalid settings.

        Port and pool size must be integers, pool size at least 1, and
        connect timeout finite and positive. Plain TCP reads the optional
        password when set. Unset optional passwords are allowed. A set instance connection name
        ignores the optional password, so a plain-TCP password never reaches the connector's IAM login.
        """
        try:
            user = os.environ[user_var]
        except KeyError:
            raise cls.config_error(f"missing required env var {user_var}") from None
        password = None
        if password_var is not None:
            try:
                password = os.environ[password_var]
            except KeyError:
                raise cls.config_error(f"missing required env var {password_var}") from None
        try:
            settings = cls(user=user, password=password, enable_iam_auth=enable_iam_auth)
        except ValidationError as exc:
            env_prefix = cls.model_config.get("env_prefix", "")
            raise cls.config_error(_validation_error_message(exc, env_prefix=env_prefix)) from exc
        if (
            password_var is None
            and optional_password_var is not None
            and settings.instance_connection_name is None
        ):
            settings = settings.model_copy(
                update={"password": os.environ.get(optional_password_var)}
            )
        return settings

    def connect_callable(self, *, connector: object | None) -> Callable[[], DBAPIConnection]:
        """A zero-arg callable suitable for `ConnectionPool(connect=...)` or
        `migrate.apply_pending(connect=...)`.
        """

        def _connect() -> DBAPIConnection:
            return connect(
                connector=connector,
                instance_connection_name=self.instance_connection_name,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                enable_iam_auth=self.enable_iam_auth,
                timeout=self.connect_timeout,
            )

        return _connect


class ConnectionSettings(BaseConnectionSettings):
    """`BaseConnectionSettings` under the `DATASET_STORE_` prefix.

    `pool_size` defaults to 10. It is smaller than the threads that can
    borrow: one `datasets_api` instance runs a read executor
    (`DatasetsBuildSettings.read_max_concurrency`, default 8) and a work
    executor (`work_max_concurrency`, 4), 12 threads, beside the
    presentation surface's reads. The default holds because a borrow lasts
    one read or one write transaction and a borrower that finds the pool
    empty waits: `ConnectionPool.borrow` waits up to 30 seconds for a
    connection, then raises `PoolTimeoutError`, which doors answer as a
    retryable 503. A refresh waiting for a materialise slot or a shared
    rebuild holds no connection, and at most `materialise_max_concurrency`
    (2) rebuilds write at once, each holding up to two. Postgres allows
    about 15 connections per tenant (`dataset_store.pool`), so a larger
    default would crowd other pools and instances. See `datasets_api.serve`'s
    module docstring for the write path's borrow arithmetic.
    """

    model_config = SettingsConfigDict(env_prefix="DATASET_STORE_", frozen=True)

    config_error: ClassVar[type[Exception]] = DatasetStoreConfigError

    database: str = "datasets"
    pool_size: int = Field(default=10, ge=1)
